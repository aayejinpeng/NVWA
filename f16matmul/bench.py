import os
import subprocess
import time
import numpy as np
from cffi import FFI
import platform
import random
import math

#TODO: Update the function definitions to reflect the new OPSNAME
OPSNAME="f16_matmul"

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
            cmd = f"gcc {cfg['flags']} -fPIC -shared  -o {cfg['out']} {cfg['src']}"
        else:
            cmd = f"gcc {cfg['flags']} -fPIC -shared -march=rv64gcv_zfh -mabi=lp64d -o {cfg['out']} {cfg['src']}"
        cmd += " -lm"
        print(f"[{name}] Building: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"[{name}] -> {cfg['out']}")

# --------------------------
# 2. 加载动态库
# --------------------------
ffi = FFI()
#TODO: Update the function definitions to reflect new OPS Input,Output,dim
ffi.cdef("void " + OPSNAME + "(void* A, void* B, float* output, int M, int N,int K);")

def load_libs():
    libs = {}
    for name, cfg in LIB_TARGETS.items():
        if os.path.isfile(cfg["out"]):
            libs[name] = ffi.dlopen(cfg["out"])
    return libs

#TODO: Update the function definitions to reflect new OPS Input,Output,dim
class ops_C:
    def __init__(self, lib, ffi, A, B): #[M,N,K]
        self.lib = lib
        self.ffi = ffi
        self.A = A
        self.B = B
        self.M, self.K = A.shape
        self.N = B.shape[0]
        self.out = np.empty((self.M, self.N), dtype=np.float32)

    def __call__(self):
        A_ptr = self.ffi.cast("void*", self.ffi.from_buffer(self.A))
        B_ptr = self.ffi.cast("void*", self.ffi.from_buffer(self.B))
        out_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.out))
        self.lib.__getattr__(OPSNAME)(A_ptr, B_ptr, out_ptr, self.M, self.N, self.K)
        return self.out

# --------------------------
# 3. Python 实现
# --------------------------
def ops_python(A,B):
    
    return ops_numpy(A,B)


# --------------------------
# 4. NumPy 实现
# --------------------------
import numpy as np


def ops_numpy(A,B):
    
    assert A.dtype == np.float16 and B.dtype == np.float16
    
    out = (A.astype(np.float32) @ (B.astype(np.float32)).T).astype(np.float32)

    out = out.astype(np.float32)
    
    return out


# --------------------------
# 5. Benchmark工具
# --------------------------
def benchmark(func, repeat=50):
    start = time.perf_counter()
    for _ in range(repeat):
        y = func()
    end = time.perf_counter()
    return (end - start) / repeat, y

def run_test(A, B, libs, repeat=50, diff_threshold=1e-5):
    print(f"\n===== Benchmark =====")

    # Python
    t_py, y_py = benchmark(lambda: ops_python(A, B), repeat)
    print(f"[Python-native] time={t_py*1e3:.3f} ms")

    # NumPy
    t_np, y_np = benchmark(lambda: ops_numpy(A, B), repeat)
    diff_np = np.max(np.abs(y_py - y_np))
    diff_np_radio = diff_np / (np.max(np.abs(y_py)) + 1e-6)
    print(f"[NumPy] time={t_np*1e3:.3f} ms  diff={diff_np:.2e}  diff_ratio={diff_np_radio:.2e}")

    #刷新输出缓冲区
    os.sys.stdout.flush()
    # C实现
    for name, lib in libs.items():
        #TODO: Update the function definitions to reflect new OPS Input,Output,dim
        func = ops_C(lib, ffi, A, B)
        t, y_c = benchmark(func, repeat)
        diff = np.max(np.abs(y_py - y_c))
        diff_radio = diff / (np.max(np.abs(y_py)) + 1e-6)
        print(f"[{name}] time={t*1e3:.3f} ms  diff={diff:.2e}  diff_ratio={diff_radio:.2e}")
        if diff_radio > diff_threshold:
            print(f"\033[31mWARNING: {name} diff {diff:.2e} exceeds threshold {diff_threshold}\033[0m")
            print(f"Difference: {diff}")
            print(f"Threshold: {diff_threshold}")
            print(f"y_py: {y_py}")
            print(f"y_c: {y_c}")
        else:
            print(f"\033[32m[{name}] Result correct within threshold {diff_threshold}\033[0m")

# --------------------------
# 6. 主入口
# --------------------------
if __name__ == "__main__":
    build_libs()
    libs = load_libs()

    #A[M,K],B[N,K],scale_A[1],scale_B[1]
    M = N = K = 256
    A = np.random.rand(M, K).astype(np.float16) * 100
    B = np.random.rand(N, K).astype(np.float16) * 100
    
    run_test(A, B, libs, repeat=1)
