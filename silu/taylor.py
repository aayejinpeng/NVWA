import numpy as np
import matplotlib.pyplot as plt
import math

# 定义原始函数
def f(x):
    return x / (1 + np.exp(-x)) # SiLU激活函数

# 定义泰勒展开函数（在x=0展开到n阶）
def taylor_silu(x, n):
    result = 0
    
    return result

# x 取值范围
x = np.linspace(-2*np.pi, 2*np.pi, 400)

# 原始函数
y = f(x)

# 泰勒展开近似（1阶、3阶、5阶）
# y1 = taylor_silu(x, 1)
# y3 = taylor_silu(x, 3)
y5 = taylor_silu(x, 5)

# 绘图
plt.figure(figsize=(8,6))
plt.plot(x, y, label='sin(x)', color='black', linewidth=2)
# plt.plot(x, y1, label='Taylor level1', linestyle='--')
# plt.plot(x, y3, label='Taylor level3', linestyle='-.')
plt.plot(x, y5, label='Taylor level5', linestyle=':')

plt.title('sin(x) and its Taylor Series Approximations')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
# plt.show()
plt.savefig("taylor_plot.png")  # 保存为图片
