import os
import subprocess
import time
import numpy as np
from cffi import FFI
import random
import math

#TODO: Update the function definitions to reflect the new OPSNAME
OPSNAME="GeLu"

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
ffi.cdef("void " + OPSNAME + "(float* input, float* output, int batch, int exhidden_dim, int hidden_dim);")

def load_libs():
    libs = {}
    for name, cfg in LIB_TARGETS.items():
        if os.path.isfile(cfg["out"]):
            libs[name] = ffi.dlopen(cfg["out"])
    return libs

#TODO: Update the function definitions to reflect new OPS Input,Output,dim
class ops_C:
    def __init__(self, lib, ffi, tensor): #[batch,exhidden_dim,hidden_dim]
        self.lib = lib
        self.ffi = ffi
        self.tensor = tensor
        self.batch, self.exhidden_dim, self.hidden_dim = tensor.shape
        self.out = np.empty((self.batch, self.exhidden_dim, self.hidden_dim), dtype=np.float32)

    def __call__(self):
        inp_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.tensor))
        out_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.out))
        self.lib.__getattr__(OPSNAME)(inp_ptr, out_ptr, self.batch, self.exhidden_dim, self.hidden_dim)
        return self.out

# --------------------------
# 3. Python 实现
# --------------------------
def ops_python(tensor):  #[batch,exhidden_dim,hidden_dim]
    """
    使用 geky 对输入激活。
    
    :param tensor: np.ndarray, 输入张量，形状为 (batch, exhidden_dim, hidden_dim)
    :return: np.ndarray, 激活后的张量，形状与输入相同
    """
    # 获取张量的形状
    batch, exhidden_dim, hidden_dim = tensor.shape
    
    # 初始化结果张量
    out = np.zeros_like(tensor)

    # for b in range(batch):
    #     for s in range(exhidden_dim):
    #         for h in range(hidden_dim):
    #             out[b, s, h] = 0.5 * tensor[b, s, h] * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (tensor[b, s, h] + 0.044715 * tensor[b, s, h] ** 3)))
                
    
    return ops_numpy(tensor)  # Updated to return the normalized output


# --------------------------
# 4. NumPy 实现
# --------------------------
import numpy as np

def ops_numpy(tensor):  #[batch,seq_len,hidden_dim]
    
    # 初始化结果张量
    out = np.zeros_like(tensor)

    out = 0.5 * tensor * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (tensor + 0.044715 * np.power(tensor, 3))))
                
    
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

def run_test(input_tensor, libs, repeat=50, diff_threshold=1e-5):
    print(f"\n===== Benchmark =====")

    # Python
    t_py, y_py = benchmark(lambda: ops_python(input_tensor), repeat)
    print(f"[Python-native] time={t_py*1e3:.3f} ms")

    # NumPy
    t_np, y_np = benchmark(lambda: ops_numpy(input_tensor), repeat)
    diff_np = np.max(np.abs(y_py - y_np))
    print(f"[NumPy] time={t_np*1e3:.3f} ms  diff={diff_np:.2e}")

    #刷新输出缓冲区
    os.sys.stdout.flush()
    # C实现
    for name, lib in libs.items():
        #TODO: Update the function definitions to reflect new OPS Input,Output,dim
        func = ops_C(lib, ffi, input_tensor)
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

    shape = [1, 256, 128]   #[batch,seq_len,hidden_dim]
    input_tensor = np.random.rand(*shape).astype(np.float32)
    per_channle_scale = np.random.rand(shape[2]).astype(np.float32)  # 每个channle的缩放因子,用于smoothquant
    run_test(input_tensor, libs, repeat=50)
