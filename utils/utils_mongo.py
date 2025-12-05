from app.schemas import schemas

def doc_to_person_read(doc: object) -> schemas.PersonRead:
    return schemas.PersonRead(
        id=str(doc["_id"]),
        chinese_name=doc.get("chinese_name",""),
        description=doc.get("description"),
        photo_path=doc.get("photo_path"),
    )
