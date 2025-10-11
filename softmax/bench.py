import os
import platform
import subprocess
import time
import numpy as np
from cffi import FFI
import random
import math

# --------------------------
# 1. 构建动态库
# --------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OPS_DIR = os.path.join(BASE_DIR)
BUILD_DIR = os.path.join(BASE_DIR, ".build")
os.makedirs(BUILD_DIR, exist_ok=True)

LIB_TARGETS = {
    "C_O0": {"src": os.path.join(OPS_DIR, "c.c"), "out": os.path.join(BUILD_DIR, "libops_o0.so"), "flags": "-O0"},
    "C_O3": {"src": os.path.join(OPS_DIR, "c.c"), "out": os.path.join(BUILD_DIR, "libops_o3.so"), "flags": "-O3"},
    "C_INTR": {"src": os.path.join(OPS_DIR, "intrinsic.c"), "out": os.path.join(BUILD_DIR, "libops_intr.so"), "flags": "-O3"},
}

def get_architecture():
    """检测当前系统架构"""
    arch = platform.machine().lower()
    if arch in ['riscv64', 'riscv32', 'riscv']:
        return 'riscv'
    if arch in ['x86_64', 'amd64', 'i386', 'i686', 'x86']:
        return 'x86'
    if arch in ['aarch64', 'arm64', 'armv7l', 'armv6l']:
        return 'arm'
    
    return arch

def build_libs():
    for name, cfg in LIB_TARGETS.items():
        if not os.path.isfile(cfg["src"]):
            print(f"[{name}] SKIP (no source: {cfg['src']})")
            continue
        arch = get_architecture()
        if arch != 'riscv':
            cmd = f"gcc {cfg['flags']} -fPIC -shared -o {cfg['out']} {cfg['src']}"
        else:
            cmd = f"gcc {cfg['flags']} -fPIC -shared -march=rv64gcv -o {cfg['out']} {cfg['src']}"
        cmd += " -lm"
        print(f"[{name}] Building: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"[{name}] -> {cfg['out']}")

# --------------------------
# 2. 加载动态库
# --------------------------
ffi = FFI()
ffi.cdef("void softmax(float* input, float* output, void* bitmask_ptr, int m, int n);")

def load_libs():
    libs = {}
    for name, cfg in LIB_TARGETS.items():
        if os.path.isfile(cfg["out"]):
            libs[name] = ffi.dlopen(cfg["out"])
    return libs

class SoftmaxC:
    def __init__(self, lib, ffi, m, n, bitmask):
        self.lib = lib
        self.ffi = ffi
        self.m = m
        self.n = n
        self.bitmask=bitmask
        self.out = np.empty((m, n), dtype=np.float32)

    def __call__(self, x: np.ndarray):
        if x.dtype != np.float32 or not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x, dtype=np.float32)
        inp_ptr = self.ffi.cast("float*", self.ffi.from_buffer(x))
        out_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.out))
        bitmask_ptr = self.ffi.cast("void*", self.ffi.from_buffer(self.bitmask))
        self.lib.softmax(inp_ptr, out_ptr, bitmask_ptr, self.m, self.n)
        return self.out

# --------------------------
# 3. Python 实现
# --------------------------
def softmax_python(batch, bitmask):
    return softmax_numpy(batch, bitmask)

# --------------------------
# 4. NumPy 实现
# --------------------------
def softmax_numpy(x: np.ndarray, bitmask):
    B, N = x.shape
    y = np.empty_like(x, dtype=np.float32)
    
    for b in range(B):
        # 从 bitmask 还原出 bool mask 向量
        mask = np.zeros(N, dtype=bool)
        for j in range(N):
            word_idx, bit_idx = divmod(j, 8)
            mask[j] = (bitmask[b, word_idx] >> bit_idx) & 1
        
        # 对无效位置设为 -inf
        masked_x = np.where(mask, x[b], -np.inf)
        
        # 计算 softmax
        x_shift = masked_x - np.max(masked_x)  # 数值稳定
        e = np.exp(x_shift)
        e *= mask  # 屏蔽无效位置
        s = np.sum(e)
        y[b] = e / s
    
    return y

# --------------------------
# 5. Benchmark工具
# --------------------------
def benchmark(func, x, repeat=50):
    start = time.perf_counter()
    for _ in range(repeat):
        y = func(x)
    end = time.perf_counter()
    return (end - start) / repeat, y

def run_test(x,bitmask, libs, repeat=50, diff_threshold=1e-5):
    print(f"\n===== Benchmark =====")

    # Python
    t_py, y_py = benchmark(lambda arr: softmax_python(arr, bitmask), x, repeat)
    print(f"[Python-native] time={t_py*1e3:.3f} ms")

    # NumPy
    t_np, y_np = benchmark(lambda arr: softmax_numpy(arr, bitmask), x, repeat)
    diff_np = np.max(np.abs(y_py - y_np))
    print(f"[NumPy] time={t_np*1e3:.3f} ms  diff={diff_np:.2e}")

    # C实现
    for name, lib in libs.items():
        func = SoftmaxC(lib, ffi, x.shape[0], x.shape[1],bitmask)
        t, y_c = benchmark(func, x, repeat)
        diff = np.max(np.abs(y_py - y_c))
        print(f"[{name}] time={t*1e3:.3f} ms  diff={diff:.2e}")
        if diff > diff_threshold or math.isnan(diff):
            print(f"WARNING: {name} diff {diff:.2e} exceeds threshold {diff_threshold}")
            print("y_py:", y_py)
            print("y_c:", y_c)

# --------------------------
# 6. 主入口
# --------------------------
if __name__ == "__main__":
    build_libs()
    libs = load_libs()

    seq_len = 512
    m, n = seq_len,seq_len
    low, high = -1500.0, 1500.0
    x = np.random.uniform(low, high, (m, n)).astype(np.float32)
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.uint8))  # 0/1 mask
    # 按位打包，每8个元素压成一个uint8
    bitmask = np.packbits(mask, axis=-1, bitorder='little')
    
    run_test(x,bitmask, libs, repeat=1)
