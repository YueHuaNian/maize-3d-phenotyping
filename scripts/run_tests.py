"""工程回归测试统一入口（Python 单元测试 + 查看器前端脚本逻辑检查）。

功能说明：
    1. 自动发现并运行 tests/ 目录下全部 Python 单元测试（建模、表型、分类、验证、查看器）；
    2. 临时测试目录固定在 .cache/test_tmp/，保证工作区根目录干净；
    3. 若本机存在 Node.js，同步运行 tests/test_viewer_ui.js 验证查看器交互逻辑（不启动浏览器）。

常用命令：
    python scripts/run_tests.py
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

# 配置项目源码路径与缓存临时目录
ROOT = Path(__file__).resolve().parents[1]
# 将运行生成的字节码集中到缓存区，源码目录保持整洁。
sys.pycache_prefix = str(ROOT / ".cache" / "python")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
tmp = ROOT / ".cache" / "test_tmp"
tmp.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(tmp)

def main():
    """依次执行 Python 与前端测试，并返回首个失败状态。"""
    # 第一步：运行全部 Python 单元测试。
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1

    # 第二步：若本机安装了 Node.js，则运行查看器前端逻辑回归测试。
    viewer_js_test = ROOT / "tests" / "test_viewer_ui.js"
    node_bin = shutil.which("node")
    if viewer_js_test.is_file() and node_bin:
        print("\n运行查看器脚本逻辑检查 (tests/test_viewer_ui.js)", flush=True)
        return subprocess.run([node_bin, str(viewer_js_test)], cwd=str(ROOT)).returncode
    print("未发现 Node.js，已跳过前端逻辑检查。", flush=True)
    return 0


if __name__ == "__main__":
    # Windows spawn 会重新导入主文件；入口保护防止子进程再次运行整套测试。
    raise SystemExit(main())
