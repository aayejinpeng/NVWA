import os
import platform
import subprocess
import time
import numpy as np
from cffi import FFI
import random
import math

#TODO: Update the function definitions to reflect the new OPSNAME
OPSNAME="fuse_shift_scale_A_right_scale_1_resadd_relu"

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
#TODO: Update the function definitions to reflect new OPS Input,Output,dim
ffi.cdef("void " + OPSNAME + "(int32_t * input, int8_t * output, uint64_t stride_input, uint64_t stride_output,uint64_t shift_scale, uint64_t dim_I,int8_t *residual);")

def load_libs():
    libs = {}
    for name, cfg in LIB_TARGETS.items():
        if os.path.isfile(cfg["out"]):
            libs[name] = ffi.dlopen(cfg["out"])
    return libs

#TODO: Update the function definitions to reflect new OPS Input,Output,dim
class ops_C:
    def __init__(self, lib, ffi, input,stride_input,stride_output,shift_scale,dim_I,residual):
        self.lib = lib
        self.ffi = ffi
        self.input = input
        self.stride_input = stride_input
        self.stride_output = stride_output
        self.shift_scale = shift_scale
        self.dim_I = dim_I
        self.residual = residual
        self.out = np.empty(input.shape, dtype=np.int8)

    def __call__(self, x: np.ndarray):
        inp_ptr = self.ffi.cast("int32_t *", self.ffi.from_buffer(self.input))
        out_ptr = self.ffi.cast("int8_t *", self.ffi.from_buffer(self.out))
        residual_ptr = self.ffi.cast("int8_t *", self.ffi.from_buffer(self.residual))
        self.lib.__getattr__(OPSNAME)(inp_ptr, out_ptr, self.stride_input, self.stride_output, self.shift_scale, self.dim_I, residual_ptr)
        return self.out

# --------------------------
# 3. Python 实现
# --------------------------

def ops_python(input,stride_input,stride_output,shift_scale,dim_I,residual): 
    # Implement the fused operation in pure Python
    # input 一定是 m×(stride_input/4)的int32数组，只处理dim_Ix64的部分
    # output 是 m×stride_output的int8数组，只修改dim_Ix64的部分
    # 
    # 计算就是对input进行shift_scale的右移(定点小数除法，对于rvv来说就是vnclip)，然后加上residual，最后relu截断到[0,127]
    
    out = np.zeros((input.shape[0], stride_output), dtype=np.int8)
    for i in range(dim_I):
        for j in range(64):
            val = ((input[i, j] + (1 << (shift_scale - 1))) >> shift_scale)
            # print(f"Debug val after first shift: {val}")
            val = ((val + (1 << (1 - 1))) >> 1)
            # print(f"Debug val after second shift: {val}")
            val = (val + residual[i, j])
            val = max(0, min(127, val))  # ReLU and clamp to [0, 127]
            out[i, j] = np.int8(val)
    
    return out
    
    
    # return out
    return out



# --------------------------
# 4. NumPy 实现
# --------------------------
import numpy as np

def ops_numpy(input,stride_input,stride_output,shift_scale,dim_I,residual): 
    # Implement the fused operation in pure Python
    # input 一定是 m×(stride_input/4)的int32数组，只处理dim_Ix64的部分
    # output 是 m×stride_output的int8数组，只修改dim_Ix64的部分
    # 
    # 计算就是对input进行shift_scale的右移(定点小数除法，对于rvv来说就是vnclip)，然后加上residual，最后relu截断到[0,127]
    
    out = np.zeros((input.shape[0], stride_output), dtype=np.int8)
    for i in range(dim_I):
        for j in range(64):
            val = ((input[i, j] + (1 << (shift_scale - 1))) >> shift_scale)
            val = ((val + (1 << (1 - 1))) >> 1)
            val = (val + residual[i, j]).astype(np.int8)
            val = max(0, min(127, val))  # ReLU and clamp to [0, 127]
            out[i, j] = np.int8(val)
        
    
    
    # return out
    return out


# --------------------------
# 5. Benchmark工具
# --------------------------
def benchmark(func, x, repeat=50):
    start = time.perf_counter()
    for _ in range(repeat):
        y = func(x)
    end = time.perf_counter()
    return (end - start) / repeat, y

def run_test(input,stride_input,stride_output,shift_scale,dim_I,residual, libs, repeat=50, diff_threshold=1e-5):
    print(f"\n===== Benchmark =====")

    # Python
    t_py, y_py = benchmark(lambda b: ops_python(b, stride_input, stride_output, shift_scale, dim_I, residual), input, repeat)
    print(f"[Python-native] time={t_py*1e3:.3f} ms")
    # return 

    # NumPy
    t_np, y_np = benchmark(lambda b: ops_numpy(b, stride_input, stride_output, shift_scale, dim_I, residual), input, repeat)
    diff_np = np.max(np.abs(y_py - y_np))
    print(f"[NumPy] time={t_np*1e3:.3f} ms  diff={diff_np:.2e}")
    

    #刷新输出缓冲区
    os.sys.stdout.flush()
    # C实现
    for name, lib in libs.items():
        #TODO: Update the function definitions to reflect new OPS Input,Output,dim
        func = ops_C(lib, ffi, input, stride_input, stride_output, shift_scale, dim_I, residual)
        t, y_c = benchmark(func, repeat)
        diff = np.max(np.abs(y_py - y_c))
        print(f"[{name}] time={t*1e3:.3f} ms  diff={diff:.2e}")
        if diff > diff_threshold:
            print(f"\033[31mWARNING: {name} diff {diff:.2e} exceeds threshold {diff_threshold}\033[0m")
            print(f"Difference: {diff}")
            print(f"Threshold: {diff_threshold}")
            #如果有错误，就输出位置和错误的值
            for i in range(y_py.shape[0]):
                for j in range(y_py.shape[1]):
                    if abs(y_py[i,j] - y_c[i,j]) > diff_threshold:
                        print(f"Mismatch at position ({i}, {j}),x={input[i,j]},res={residual[i,j]}: Python={y_py[i,j]}, C={y_c[i,j]}")
        else:
            print(f"\033[32m[{name}] Result correct within threshold {diff_threshold}\033[0m")

# --------------------------
# 6. 主入口
# --------------------------
if __name__ == "__main__":
    build_libs()
    libs = load_libs()
    
    # 准备输入数据，-2^15~+2^15之间的整数
    input_tensor = np.random.randint(-32768, 32767, size=(128, 256), dtype=np.int32)
    stride_input = 256*4
    stride_output = 256
    shift_scale = 9
    dim_I = 49
    residual = np.random.randint(0, 16, size=(128, 256), dtype=np.int8)

    run_test(input_tensor, stride_input, stride_output, shift_scale, dim_I, residual, libs, repeat=1, diff_threshold=1e-5)
