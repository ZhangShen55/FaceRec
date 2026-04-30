"""
Cython 编译后清理 —— 删除 .py / .c 中间产物，只保留 .so

执行时机：在 Dockerfile builder 阶段 `python build_cython.py build_ext --inplace` 之后

清理规则：
- 编译目录下若 .py 已有同名 .so → 删除 .py（防止源码进入 runtime 镜像）
- 编译目录下所有 .c 文件（Cython 中间产物）→ 删除
- 编译目录下 build/ 目录 → 删除
- 始终保留所有 __init__.py（不论是否有 .so）
- 不动 models/、main.py 等明确不编译的位置
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

COMPILE_DIRS = ["core", "services", "router", "middleware", "utils"]
KEEP_PY = {"__init__.py"}


def main():
    removed_py = 0
    removed_c = 0
    removed_build_dirs = 0

    for d in COMPILE_DIRS:
        base = ROOT / d
        if not base.exists():
            continue

        # 1) 删除有对应 .so 的 .py
        for py in base.rglob("*.py"):
            if py.name in KEEP_PY:
                continue
            so_candidates = list(py.parent.glob(f"{py.stem}.cpython-*.so"))
            if so_candidates:
                py.unlink()
                removed_py += 1
                print(f"[clean] del .py  {py.relative_to(ROOT)}")

        # 2) 删除 .c 中间文件
        for c in base.rglob("*.c"):
            c.unlink()
            removed_c += 1
            print(f"[clean] del .c   {c.relative_to(ROOT)}")

    # 3) 删除根目录下的 build/（setup 产物）
    build_dir = ROOT / "build"
    if build_dir.exists() and build_dir.is_dir():
        shutil.rmtree(build_dir)
        removed_build_dirs += 1
        print(f"[clean] rmtree   build/")

    print(
        f"[clean] ✅ 清理完成: 删除 .py {removed_py} 个 / "
        f".c {removed_c} 个 / build/ {removed_build_dirs} 个"
    )


if __name__ == "__main__":
    main()
