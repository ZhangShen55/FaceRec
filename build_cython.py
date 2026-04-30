"""
Cython 编译脚本 —— 把业务代码编译为 .so（加密）

策略：
- 编译: core/ services/ router/ middleware/ utils/  → .cpython-310-x86_64-linux-gnu.so
- 不编译: main.py / models/ / core/config.py / 所有 __init__.py
  原因：
    - main.py: uvicorn 反射启动入口
    - models/: Pydantic v2 内部反射 + field_validator,Cython 后兼容性差
    - core/config.py: Pydantic v2 BaseModel + computed_field，Cython 后会丢失反射信息
    - __init__.py: 保留以维持包结构

用法（在容器 builder 阶段执行）:
    cd /build/app
    python build_cython.py build_ext --inplace -j 4

注意：
    本文件位于 /build/app/build_cython.py，但 setup() 必须从包父目录 /build
    执行。否则 Cython 会把模块名识别成 app.core.xxx，却尝试复制到
    /build/app/app/core，导致 "No such file or directory"。
"""
import os
import sys
from pathlib import Path
from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options

# 关闭 docstring/位置注释，进一步减少 .so 中可读字符串
Options.docstrings = False
Options.embed_pos_in_docstring = False

ROOT = Path(__file__).resolve().parent          # /build/app
PROJECT_ROOT = ROOT.parent                      # /build
PACKAGE_NAME = ROOT.name                        # app

# 需要编译为 .so 的目录（业务逻辑/算法）
COMPILE_DIRS = [
    "core",
    "services",
    "router",
    "middleware",
    "utils",
]

# 这些文件即便在 COMPILE_DIRS 内也不编译
EXCLUDE_FILES = {
    "__init__.py",                # 保留所有包入口
}

# 这些具体路径即便在 COMPILE_DIRS 内也不编译（相对 app/）
EXCLUDE_PATHS = {
    "core/config.py",             # Pydantic 配置模型，必须保留源码供反射
}

# 这些子目录无论如何都不编译
EXCLUDE_DIRS_GLOB = {
    "__pycache__",
}


def collect_py_files() -> list[str]:
    """收集所有需要编译的 .py 文件（相对包父目录 /build）"""
    files = []
    for d in COMPILE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for py in base.rglob("*.py"):
            # 跳过排除目录
            if any(part in EXCLUDE_DIRS_GLOB for part in py.parts):
                continue
            # 跳过排除文件
            if py.name in EXCLUDE_FILES:
                continue
            # 跳过排除路径
            rel_to_root = py.relative_to(ROOT).as_posix()
            if rel_to_root in EXCLUDE_PATHS:
                continue
            files.append(str(py.relative_to(PROJECT_ROOT)))
    return files


def main():
    py_files = collect_py_files()
    print(f"[Cython] 包目录: {ROOT}")
    print(f"[Cython] setup 工作目录: {PROJECT_ROOT}")
    print(f"[Cython] 待编译文件数: {len(py_files)}")
    for f in py_files:
        display = Path(f)
        if display.parts and display.parts[0] == PACKAGE_NAME:
            display = Path(*display.parts[1:])
        print(f"  - {display}")

    if not py_files:
        print("[Cython] 没有可编译文件,退出")
        sys.exit(0)

    # 关键：从包父目录执行 setup，保证 build_ext --inplace 把 app.core.*
    # 复制回 /build/app/core，而不是 /build/app/app/core。
    os.chdir(PROJECT_ROOT)

    setup(
        name="facerecapi_compiled",
        script_args=sys.argv[1:] or ["build_ext", "--inplace"],
        options={"build": {"build_base": str(ROOT / "build")}},
        ext_modules=cythonize(
            py_files,
            compiler_directives={
                "language_level": "3",
                # FastAPI 路由装饰器需要保留关键字参数与函数签名
                "always_allow_keywords": True,
                "binding": True,
                "embedsignature": False,
                "linetrace": False,
                # 性能优化:关闭边界检查（业务代码不依赖这些）
                "boundscheck": False,
                "wraparound": False,
                "initializedcheck": False,
            },
            nthreads=int(os.environ.get("CYTHON_NTHREADS", "4")),
            quiet=False,
        ),
        zip_safe=False,
    )

    print(f"[Cython] ✅ 编译完成,共 {len(py_files)} 个模块")


if __name__ == "__main__":
    main()
