import os
import subprocess
import time
import numpy as np
from cffi import FFI
import platform
import random
import math

#TODO: Update the function definitions to reflect the new OPSNAME
OPSNAME="smoothquantO1"

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
            cmd = f"gcc {cfg['flags']} -fPIC -shared -march=rv64gcv -mabi=lp64d -o {cfg['out']} {cfg['src']}"
        cmd += " -lm"
        print(f"[{name}] Building: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"[{name}] -> {cfg['out']}")

# --------------------------
# 2. 加载动态库
# --------------------------
ffi = FFI()
#TODO: Update the function definitions to reflect new OPS Input,Output,dim
ffi.cdef("void " + OPSNAME + "(float* A, int8_t* output,float* scale, int M,int K);")

def load_libs():
    libs = {}
    for name, cfg in LIB_TARGETS.items():
        if os.path.isfile(cfg["out"]):
            libs[name] = ffi.dlopen(cfg["out"])
    return libs

#TODO: Update the function definitions to reflect new OPS Input,Output,dim
class ops_C:
    def __init__(self, lib, ffi, A): #[M,N,K]
        self.lib = lib
        self.ffi = ffi
        self.A = A
        self.M, self.K = A.shape
        self.out = np.empty((self.M, self.K), dtype=np.int8)
        self.out_scale = np.empty((self.M,), dtype=np.float32)

    def __call__(self):
        A_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.A))
        out_ptr = self.ffi.cast("int8_t*", self.ffi.from_buffer(self.out))
        out_scale_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.out_scale))
        self.lib.__getattr__(OPSNAME)(A_ptr, out_ptr, out_scale_ptr,self.M, self.K)
        return self.out,self.out_scale

# --------------------------
# 3. Python 实现
# --------------------------
def ops_python(A):  #[M,K],[N,K],[1]
    """
    使用 RMSNorm 对输入张量进行归一化。
    
    :param A: np.ndarray, 输入张量 A,形状为 (M, K)
    :param B: np.ndarray, 输入张量 B,形状为 (N, K)
    :param scale_A: np.ndarray, A 的缩放因子，是自己量化出来的！
    :param scale_B: np.ndarray, B 的缩放因子，形状为 (1,)
    :return: np.ndarray, 归一化后的张量，形状与输入相同
    """
    assert A.dtype == np.float32
    # # 获取张量的形状
    # M,K = A.shape
    # N = B.shape[0]
    
    # # 初始化结果张量
    # out = np.zeros((M,N), dtype=np.float32)
    # for m in range(M):
    #     for n in range(N):
    #         sum = 0.0
    #         for k in range(K):
    #             sum += (A[m,k]) * (B[n,k])
    #         out[m,n] = sum
    # out = out * scale_A[0] * scale_B[0]
    
    return ops_numpy(A)  # Updated to return the normalized output


# --------------------------
# 4. NumPy 实现
# --------------------------
import numpy as np



#w8a8matmul_smoothquant01,A是在线量化得到的
def ops_numpy(A):  #[M,K]
    
    assert A.dtype == np.float32
    
    # 获取张量的形状
    M,K = A.shape
    # 初始化结果张量
    # out = np.zeros((M,N), dtype=np.float32)
    
    #先对A完成per token的量化到I8
    #求张量A每个token的最大值
    scales = np.max(np.abs(A),axis=1,keepdims=True)  #[M,1]
    # print(A)
    # print(scales)
    q8_max = 127
    scales = scales.clip(min=1e-5) / q8_max
    q_A = (A / scales).round().clip(-128,127).astype(np.int8)
    
    return q_A,scales


# --------------------------
# 5. Benchmark工具
# --------------------------
def benchmark(func, repeat=50):
    start = time.perf_counter()
    for _ in range(repeat):
        y = func()
    end = time.perf_counter()
    return (end - start) / repeat, y

def run_test(A,libs, repeat=50, diff_threshold=1e-5):
    print(f"\n===== Benchmark =====")

    # Python
    t_py, y_py = benchmark(lambda: ops_python(A), repeat)
    print(f"[Python-native] time={t_py*1e3:.3f} ms")

    # NumPy
    t_np, y_np = benchmark(lambda: ops_numpy(A), repeat)
    diff_np = np.max(np.abs(y_py[0] - y_np[0]))
    print(f"[NumPy] time={t_np*1e3:.3f} ms  diff={diff_np:.2e}")

    #刷新输出缓冲区
    os.sys.stdout.flush()
    # C实现
    for name, lib in libs.items():
        #TODO: Update the function definitions to reflect new OPS Input,Output,dim
        func = ops_C(lib, ffi, A)
        t, y_c = benchmark(func, repeat)
        diff = np.max(np.abs(y_py[0] - y_c[0]))
        print(f"[{name}] time={t*1e3:.3f} ms  diff={diff:.2e}")
        if diff > diff_threshold:
            print(f"\033[31mWARNING: {name} diff {diff:.2e} exceeds threshold {diff_threshold}\033[0m")
            print(f"Difference: {diff}")
            print(f"Threshold: {diff_threshold}")
            print(f"y_py: {y_py[0]}")
            print(f"y_c: {y_c[0]}")
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
    A = (np.random.rand(M, K).astype(np.float32) - np.random.rand(M, K).astype(np.float32)) * 1000
    
    
    run_test(A, libs, repeat=1)
