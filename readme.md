Hello there!
This project is NVWA(Nimble Vector Workbench for AI)


新手教程（中文版）：

1. 配置环境

``` bash
# path/NVWA/
./initenv.sh
```

2. 加载配置的python环境
``` bash
# path/NVWA/
source .venv/bin/activate
```

3. 算子文件夹中的文件
以`path/NVWA/softmax`表示的softmax算子为例：
* `c.c`：自己写的算子的C语言实现
* `intrinsic.c`:自己写的不同架构的算子intrinsic函数实现
* `bench.py`：包含python的实现的算子的性能和正确性脚本
    + 把上面的.c们编译成动态库
    + 将动态库中的算子函数执行很多遍
    + 也会执行python自己的算子实现
    + 最终输出各形式算子的执行时间
``` bash
# path/NVWA/softmax
python bench.py
# 输出结果
[Python-native] time=10.546 ms
[NumPy] time=0.200 ms  diff=3.49e-10
[C_O0] time=0.912 ms  diff=2.21e-09
[C_O3] time=0.548 ms  diff=2.21e-09
[C_INTR] time=0.389 ms  diff=2.21e-09
```
4. 在任意远程的开发板上执行
``` bash
# 在任意远程的开发板上执行bench.py
./ test-remote.sh -a
```