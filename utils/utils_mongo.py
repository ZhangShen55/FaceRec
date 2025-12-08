from app.schemas import schemas

def doc_to_person_read(doc: object) -> schemas.PersonRead:
    return schemas.PersonRead(
        id=str(doc["_id"]),
        name=doc.get("name"),
        number=doc.get("number"),
        photo_path=doc.get("photo_path"),
    )
