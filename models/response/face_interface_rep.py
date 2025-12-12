from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class BBox(BaseModel):
    # 人脸位置
    x: int
    y: int
    w: int
    h: int

class MatchItem(BaseModel):
    reason: str
    id: str
    name: Optional[str] = None
    number: Optional[str] = None
    similarity: str

class RecognizeResp(BaseModel):
    has_face: bool
    bbox: Optional[BBox] = None
    threshold: float
    match: Optional[MatchItem] = None
    message: str

    # 工厂方法
    @classmethod
    def from_recognize(
        cls,
        *,
        matched: bool,
        has_face: bool,
        bbox: Optional[Dict[str, Any]] = None,
        best_sim: float,
        threshold: float,
        reason: str = "",
        person_doc: Optional[Dict[str, Any]] = None,
    ) -> "RecognizeResp":
        """统一构造器，业务层直接 RecognizeResp.from_recognize(...) 即可"""
        if not has_face:
            return cls(
                has_face=False,
                bbox=None,
                threshold=threshold,
                match=None,
                message="图像中未检测到人脸，请重新捕捉人脸",
            )

        if not matched:
            return cls(
                has_face=True,
                bbox=BBox(**bbox) if bbox else None,
                threshold=threshold,
                match=None,
                message="匹配失败，未能够匹配到目标人物",
            )

        if best_sim >= threshold:
            return cls(
                has_face=True,
                bbox=BBox(**bbox) if bbox else None,
                threshold=threshold,
                match=MatchItem(
                    reason=reason,
                    id=str(person_doc["_id"]),
                    name=person_doc.get("name"),
                    number=person_doc.get("number"),
                    similarity=f"{best_sim * 100:.2f}%",
                ),
                message=f"匹配成功，检测相似度:{best_sim * 100:.2f}% >= 阈值{threshold * 100:.2f}%",
            )

        return cls(
            has_face=True,
            bbox=BBox(**bbox) if bbox else None,
            threshold=threshold,
            match=None,
            message=f"匹配失败，检测相似度:{best_sim * 100:.2f}% < 阈值{threshold * 100:.2f}%",
        )
