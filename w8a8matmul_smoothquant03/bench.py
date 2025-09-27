import os
import subprocess
import time
import numpy as np
from cffi import FFI
import random
import math

#TODO: Update the function definitions to reflect the new OPSNAME
OPSNAME="w8a8matmul_smoothquant03"

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

def build_libs():
    for name, cfg in LIB_TARGETS.items():
        if not os.path.isfile(cfg["src"]):
            print(f"[{name}] SKIP (no source: {cfg['src']})")
            continue
        cmd = f"gcc {cfg['flags']} -fPIC -shared -o {cfg['out']} {cfg['src']} -lm"
        print(f"[{name}] Building: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        print(f"[{name}] -> {cfg['out']}")

# --------------------------
# 2. 加载动态库
# --------------------------
ffi = FFI()
#TODO: Update the function definitions to reflect new OPS Input,Output,dim
ffi.cdef("void " + OPSNAME + "(int8_t* A, int8_t* B,float* A_scale, float* B_scale, float* output, int M, int N,int K);")

def load_libs():
    libs = {}
    for name, cfg in LIB_TARGETS.items():
        if os.path.isfile(cfg["out"]):
            libs[name] = ffi.dlopen(cfg["out"])
    return libs

#TODO: Update the function definitions to reflect new OPS Input,Output,dim
class ops_C:
    def __init__(self, lib, ffi, A, B, A_scale, B_scale): #[M,N,K]
        self.lib = lib
        self.ffi = ffi
        self.A = A
        self.B = B
        self.A_scale = A_scale
        self.B_scale = B_scale
        self.M, self.K = A.shape
        self.N = B.shape[0]
        self.out = np.empty((self.M, self.N), dtype=np.float32)

    def __call__(self):
        A_ptr = self.ffi.cast("int8_t*", self.ffi.from_buffer(self.A))
        B_ptr = self.ffi.cast("int8_t*", self.ffi.from_buffer(self.B))
        out_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.out))
        A_scale_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.A_scale))
        B_scale_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.B_scale))
        self.lib.__getattr__(OPSNAME)(A_ptr, B_ptr, A_scale_ptr, B_scale_ptr, out_ptr, self.M, self.N, self.K)
        return self.out

# --------------------------
# 3. Python 实现
# --------------------------
def ops_python(A,B,scale_A,scale_B):  #[M,K],[N,K],[1],[1]
    """
    使用 RMSNorm 对输入张量进行归一化。
    
    :param A: np.ndarray, 输入张量 A,形状为 (batch, M, K)
    :param B: np.ndarray, 输入张量 B,形状为 (batch, N, K)
    :param scale_A: np.ndarray, A 的缩放因子，形状为 (1,)
    :param scale_B: np.ndarray, B 的缩放因子，形状为 (1,)
    :return: np.ndarray, 归一化后的张量，形状与输入相同
    """
    assert A.dtype == np.int8 and B.dtype == np.int8
    assert scale_A.shape == (1,) and scale_B.shape == (1,)
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
    
    return ops_numpy(A,B,scale_A,scale_B)  # Updated to return the normalized output


# --------------------------
# 4. NumPy 实现
# --------------------------
import numpy as np

def ops_numpy(A,B,scale_A,scale_B):  #[M,K],[N,K],[1],[1]
    
    assert A.dtype == np.int8 and B.dtype == np.int8
    assert scale_A.shape == (1,) and scale_B.shape == (1,)
    
    # 获取张量的形状
    M,K = A.shape
    N,_ = B.shape
    # 初始化结果张量
    # out = np.zeros((M,N), dtype=np.float32)
    
    out = (A.astype(np.int32) @ (B.astype(np.int32)).T).astype(np.int32)
    out = out.astype(np.float32) 
    out = out * scale_A[0] * scale_B[0]
    
    #numpy矩阵直接矩阵乘法
    

    out = out.astype(np.float32)
    
    return out  # Updated to return the normalized output


# --------------------------
# 5. Benchmark工具
# --------------------------
def benchmark(func, repeat=50):
    start = time.perf_counter()
    for _ in range(repeat):
        y = func()
    end = time.perf_counter()
    return (end - start) / repeat, y

def run_test(A, B, A_scale, B_scale, libs, repeat=50, diff_threshold=1e-5):
    print(f"\n===== Benchmark =====")

    # Python
    t_py, y_py = benchmark(lambda: ops_python(A, B, A_scale, B_scale), repeat)
    print(f"[Python-native] time={t_py*1e3:.3f} ms")

    # NumPy
    t_np, y_np = benchmark(lambda: ops_numpy(A, B, A_scale, B_scale), repeat)
    diff_np = np.max(np.abs(y_py - y_np))
    print(f"[NumPy] time={t_np*1e3:.3f} ms  diff={diff_np:.2e}")

    #刷新输出缓冲区
    os.sys.stdout.flush()
    # C实现
    for name, lib in libs.items():
        #TODO: Update the function definitions to reflect new OPS Input,Output,dim
        func = ops_C(lib, ffi, A, B, A_scale, B_scale)
        t, y_c = benchmark(func, repeat)
        diff = np.max(np.abs(y_py - y_c))
        print(f"[{name}] time={t*1e3:.3f} ms  diff={diff:.2e}")
        if diff > diff_threshold:
            print(f"\033[31mWARNING: {name} diff {diff:.2e} exceeds threshold {diff_threshold}\033[0m")
            print(f"Difference: {diff}")
            print(f"Threshold: {diff_threshold}")
        else:
            print(f"\033[32m[{name}] Result correct within threshold {diff_threshold}\033[0m")

# --------------------------
# 6. 主入口
# --------------------------
if __name__ == "__main__":
    build_libs()
    libs = load_libs()

    #A[M,K],B[N,K],scale_A[1],scale_B[1]
    M = N = K = 4
    A = np.random.randint(-128, 127, size=(M, K), dtype=np.int8)
    B = np.random.randint(-128, 127, size=(N, K), dtype=np.int8)
    A_scale = np.random.rand(1).astype(np.float32) * 1
    B_scale = np.random.rand(1).astype(np.float32) * 1
    
    
    run_test(A, B, A_scale, B_scale, libs, repeat=50)
