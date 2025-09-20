import os
import subprocess
import time
import numpy as np
from cffi import FFI
import random
import math

#TODO: Update the function definitions to reflect the new OPSNAME
OPSNAME="rope"

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
ffi.cdef("void " + OPSNAME + "(float* input, float* output, float* rope_theta, int pos, int batch, int n_head, int seq_len, int head_dim);")

def load_libs():
    libs = {}
    for name, cfg in LIB_TARGETS.items():
        if os.path.isfile(cfg["out"]):
            libs[name] = ffi.dlopen(cfg["out"])
    return libs

#TODO: Update the function definitions to reflect new OPS Input,Output,dim
class ops_C:
    def __init__(self, lib, ffi, tensor, rope_theta, pos=0):
        self.lib = lib
        self.ffi = ffi
        self.tensor = tensor
        self.batch, self.n_head, self.seq_len, self.head_dim = tensor.shape
        self.rope_theta = rope_theta
        self.pos = pos
        self.out = np.empty((self.batch, self.n_head, self.seq_len, self.head_dim), dtype=np.float32)

    def __call__(self, x: np.ndarray):
        if x.dtype != np.float32 or not x.flags['C_CONTIGUOUS']:
            x = np.ascontiguousarray(x, dtype=np.float32)
        inp_ptr = self.ffi.cast("float*", self.ffi.from_buffer(x))
        rope_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.rope_theta))
        out_ptr = self.ffi.cast("float*", self.ffi.from_buffer(self.out))
        self.lib.__getattr__(OPSNAME)(inp_ptr, out_ptr, rope_ptr, self.pos, self.batch, self.n_head, self.seq_len, self.head_dim)
        return self.out

# --------------------------
# 3. Python 实现
# --------------------------
def get_rope_theta(head_dim, basefreq=10000): #根据head_dim计算得到一个sin(theta)和cos(theta)的两个向量
    dim = np.arange(head_dim//2)
    inv_freq = 1.0 / (basefreq ** (dim / (head_dim//2)))  # 形状 [head_dim//2]
    return inv_freq.astype(np.float32)  # 形状 [head_dim//2],get_rope_theta*pos = angle


def ops_python(tensor, rope_theta, pos = 0):  #[batch,n_head,seq_len,head_dim]
    """
    使用 ROTARY POSITION EMBEDDING (ROPE) 对张量进行位置编码。
    
    :param tensor: np.ndarray, 输入张量，形状为 [batch, n_head, seq_len, head_dim]
    :param rope_theta: np.ndarray, 预计算的旋转角度，形状为 [head_dim//2]
    :param pos: int, 位置索引, 默认为0
    :return: np.ndarray, 位置编码后的张量，形状与输入相同
    """
    # # 获取张量的形状
    # batch, n_head, seq_len, head_dim = tensor.shape
    
    # # 初始化结果张量
    # out = np.zeros_like(tensor)

    # # 遍历每个 batch 和每个 sequence
    # for b in range(batch):
    #     for i in range(n_head):
    #         for j in range(seq_len):
    #             for k in range(0, head_dim, 2):
    #                 # 计算每个位置和维度的旋转角度
    #                 pos_ = j + pos
    #                 dim_ = k
    #                 angle = rope_theta[dim_ // 2] * pos_
    #                 sin_val = np.sin(angle)
    #                 cos_val = np.cos(angle)
                    
    #                 # 计算旋转后的值
    #                 x = tensor[b, i, j, k]
    #                 y = tensor[b, i, j, k+1]
                    
    #                 # 对每个维度做旋转
    #                 out[b, i, j, k] = x * cos_val - y * sin_val
    #                 out[b, i, j, k+1] = x * sin_val + y * cos_val
    
    # return out
    return ops_numpy(tensor, rope_theta, pos)


# --------------------------
# 4. NumPy 实现
# --------------------------
import numpy as np

def ops_numpy(tensor, rope_theta, pos=0):  #[batch,n_head, seq_len, head_dim]
    out = []  #[batch, seq_len, n_head, head_dim]
    
    _, _, seq_len, _ = tensor.shape
    
    # 计算位置编码（rotary position embedding）
    positions = np.arange(seq_len) + pos  # 形状 [seq_len]

    # 计算最终的旋转角度 idx_theta
    idx_theta = positions[:, None] * rope_theta[None, :]  # 形状 [seq_len, head_dim//2]

    # 分成两部分：正弦和余弦
    sin = np.sin(idx_theta)
    cos = np.cos(idx_theta)

    # 使用正弦和余弦进行旋转编码
    for b in range(tensor.shape[0]):
        out_b = np.zeros_like(tensor[b])  # 形状 [n_head, seq_len, head_dim]
        for i in range(tensor.shape[1]):  # 遍历每个头
            x1 = tensor[b, i, :, 0::2]  # 偶数索引
            x2 = tensor[b, i, :, 1::2]  # 奇数索引
            out_left = np.zeros_like(x1)
            out_left = x1 * cos - x2 * sin
            out_right = np.zeros_like(x2)
            out_right = x1 * sin + x2 * cos
            out_b = np.concatenate([out_left, out_right], axis=-1)
        out.append(out_b)
        
    return np.array(out, dtype=np.float32)


# --------------------------
# 5. Benchmark工具
# --------------------------
def benchmark(func, x, repeat=50):
    start = time.perf_counter()
    for _ in range(repeat):
        y = func(x)
    end = time.perf_counter()
    return (end - start) / repeat, y

def run_test(input_tensor, libs, repeat=50, diff_threshold=1e-5,basefreq=10000):
    print(f"\n===== Benchmark =====")

    rope_theta = get_rope_theta(input_tensor.shape[-1], basefreq)
    # Python
    t_py, y_py = benchmark(lambda b: ops_python(b, rope_theta=rope_theta, pos=0), input_tensor, repeat)
    print(f"[Python-native] time={t_py*1e3:.3f} ms")

    # NumPy
    t_np, y_np = benchmark(lambda b: ops_numpy(b, rope_theta=rope_theta, pos=0), input_tensor, repeat)
    diff_np = np.max(np.abs(y_py - y_np))
    print(f"[NumPy] time={t_np*1e3:.3f} ms  diff={diff_np:.2e}")

    #刷新输出缓冲区
    os.sys.stdout.flush()
    # C实现
    for name, lib in libs.items():
        #TODO: Update the function definitions to reflect new OPS Input,Output,dim
        func = ops_C(lib, ffi, input_tensor, rope_theta=rope_theta, pos=0)
        t, y_c = benchmark(func, input_tensor, repeat)
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

    shape = [1, 1, 256, 128]  #[batch,n_head,seq_len,head_dim]
    input_tensor = np.random.rand(*shape).astype(np.float32)

    run_test(input_tensor, libs, repeat=50)
