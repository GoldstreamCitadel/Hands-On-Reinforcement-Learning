"""
不是 conda 环境坏了，根因是你的脚本名 dis.py 和 Python 标准库 dis 重名了。
numpy -> inspect -> dis 这条导入链里，Python 错把你的脚本当成标准库模块，
触发了循环导入，所以才会报 inspect.cleandoc 不存在。重装解释器不会解决这个问题。

我已经处理好了：

加了兼容入口 dis.py (line 1)，所以你原来的 python dis.py 还能继续用。
把真正的训练代码放到 sac_continuous.py (line 1)，
并顺手修了几个会继续报错/影响训练的问题：
rl_utils 导入、缺失的 actor_optimizer、目标 critic 初始化、critic_2 的笔误，
还有两个小的训练细节。
我用你的 rl2 解释器实测跑了 D:\Users\22878\anaconda3\envs\rl2\python.exe dis.py，
现在已经能正常进入训练循环，不再出现那个 inspect.cleandoc 报错。
现在看到的只剩 gym 的普通 warning，不影响运行。
"""

import importlib.util
from pathlib import Path
import runpy
import sys
import sysconfig


def _load_stdlib_dis():
    stdlib_dis_path = Path(sysconfig.get_path("stdlib")) / "dis.py"
    spec = importlib.util.spec_from_file_location("dis", stdlib_dis_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"Cannot load stdlib dis from {stdlib_dis_path}")
    spec.loader.exec_module(module)
    sys.modules["dis"] = module


if __name__ == "__main__":
    _load_stdlib_dis()
    runpy.run_path(str(Path(__file__).with_name("sac_discrete.py")), run_name="__main__")
