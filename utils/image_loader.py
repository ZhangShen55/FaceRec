# app/image_loader.py
from __future__ import annotations
import io, base64, re, httpx, cv2, numpy as np
from fastapi import Form, File, UploadFile, HTTPException, Depends
from typing import Union, Tuple

# 1. 先让 FastAPI 把「字符串」或「文件」都原样交给我们
async def _raw_photo(
    photo: Union[str, UploadFile] = Form(...)  # 只留一个字段
) -> Union[str, bytes]:
    """
    返回:
        str  -> URL 或 base64
        bytes-> 文件内容
    """
    if isinstance(photo, str):
        return photo          # URL 或 base64
    # 否则是 UploadFile
    return await photo.read()



async def get_photo_mat(raw: Union[str, bytes, UploadFile] = Depends(_raw_photo)) -> Tuple[np.ndarray, str]:
    """
    获取图像矩阵和文件名，支持 URL、Base64 或文件上传。
    """
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("http"):
            # URL 分支
            try:
                resp = httpx.get(raw, timeout=10, follow_redirects=True)
                resp.raise_for_status()
            except Exception as e:
                raise HTTPException(4001, f"下载图片失败: {e}")
            return _decode(resp.content), raw  # 返回 URL 和图片数据
        else:
            # base64 分支
            raw = re.sub(r'^data:image/\w+;base64,', '', raw)
            try:
                content = base64.b64decode(raw)
            except Exception as e:
                raise HTTPException(4002, f"base64 解码失败: {e}")
            return _decode(content), "base64_image"  # 返回 base64 和图片数据


    elif isinstance(raw, bytes):
        if len(raw) == 0:
            raise HTTPException(400, "接收到的二进制数据为空")
        return _decode(raw), "unknown_file"


    elif isinstance(raw, UploadFile):
        # 1. 读取文件内容
        contents = await raw.read()
        # 【新增】检查文件是否为空 (0字节)
        if len(contents) == 0:
            # 这是一个非常具体的客户端错误，明确告知哪个文件有问题
            raise HTTPException(
                status_code=400,
                detail=f"上传的文件 '{raw.filename}' 是空的 (0KB)，请检查文件是否损坏"
            )
        # 【建议】同时复用之前提到的“最大文件限制”，防止内存炸弹
        # MAX_IMAGE_SIZE = 10 * 1024 * 1024  (假设定义了常量)
        # if len(contents) > MAX_IMAGE_SIZE:
        #     raise HTTPException(status_code=400, detail="文件过大")
        return _decode(contents), raw.filename  # 返回图片数据和文件名

    # 如果没有找到匹配的情况，抛出异常
    raise HTTPException(status_code=4003, detail="无法解析图片")

# ---------- 通用解码 ----------
def _decode(content: bytes) -> np.ndarray:
    # 【修复 1】在一切开始前，检查 bytes 是否为空
    if not content:
        raise HTTPException(status_code=400, detail="上传的图片内容为空")

    nparr = np.frombuffer(content, np.uint8)

    # 【修复 2】即使 bytes 不为空，也要防止转换后的数组为空
    if nparr.size == 0:
        raise HTTPException(status_code=400, detail="图片数据无效(Size 0)")

    # 只有确保 nparr 有内容了，才敢传给 cv2
    try:
        mat = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        # 捕获 OpenCV 可能抛出的其他奇奇怪怪的错误
        raise HTTPException(status_code=400, detail=f"图片解码异常: {str(e)}")

    if mat is None:
        raise HTTPException(status_code=400, detail="图片解码失败，文件格式可能不受支持")

    # 【运维建议】强烈建议加上这一步：规整内存，防止 Dlib 崩溃
    if not mat.flags['C_CONTIGUOUS']:
        mat = np.ascontiguousarray(mat)

    return mat


# 2. 再依赖上一层，做真正的解析
# async def get_photo_mat(raw: Union[str, bytes] = Depends(_raw_photo)) -> np.ndarray:
#     if isinstance(raw, str):
#         raw = raw.strip()
#         if raw.startswith("http"):
#             # URL 分支
#             try:
#                 resp = httpx.get(raw, timeout=10, follow_redirects=True)
#                 resp.raise_for_status()
#             except Exception as e:
#                 raise HTTPException(400, f"下载图片失败: {e}")
#             return _decode(resp.content)
#         else:
#             # base64 分支
#             raw = re.sub(r'^data:image/\w+;base64,', '', raw)
#             try:
#                 content = base64.b64decode(raw)
#             except Exception as e:
#                 raise HTTPException(400, f"base64 解码失败: {e}")
#             return _decode(content)
#     # bytes 分支（文件）
#     return _decode(raw)
