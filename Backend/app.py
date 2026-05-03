from fastapi import FastAPI, Depends, HTTPException, Response, Query
from sqlmodel import Session, select, or_, func
from .models import *
from .utils import get_session, create_db

app=FastAPI()

create_db()



@app.post("/pacients",response_model=PacientPublic)
def create_pacient(pacient:PacientCreate, session=Depends(get_session)):
    db_pacient=PacientDB.model_validate(pacient)
    session.add(pacient)
    session.commit()
    session.refresh(db_pacient)
    return db_pacient

@app.get("/pacients",response_model=PacientList)
def get_pacients_list(offset:int=0, limit: int=Query(default=10, le=10),search: str|None=None, session=Depends(get_session)):
    query=select(PacientDB)
    if search:
        query=query.where(or_(
            PacientDB.pacient_name.ilike(f"%{search}%"),
            PacientDB.update_date.ilike(f"%{search}%")
        ))
    total=session.exec(select(func.count()).select_from(query.subquery())).one()
    pacients=session.exec(query.offset(offset).limit(limit)).all()
    return PacientList(items=pacients, total=total, offset=offset, limit=limit)

@app.get("/pacients/{pacient_id}", response_model=PacientPublic)
def get_pacient(pacient_id:int, session=Depends(get_session)):
    pacient=session.get(PacientDB,pacient_id)
    if not pacient:
        raise HTTPException(status_code=404, detail="Pacient not found")
    return pacient

@app.delete("/pacients/{pacient_id}")
def delete_pacient(pacient_id: int, session=Depends(get_session)):
    pacient=session.get(PacientDB, pacient_id)
    if not pacient:
            raise HTTPException(status_code=404, detail="Note not found")
    session.delete(pacient)
    session.commit()
    return Response(status_code=204)

@app.patch("/pacients/{pacient_id}", response_model=PacientPublic)
def update_pacient(pacient_id: int, update:PacientUpdate, session=Depends(get_session)):
    pacient=session.get(PacientDB,pacient_id)
    if not pacient: 
        raise HTTPException(status_code=404, detail="Note not found")

    update_data=update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(pacient, key, value)
        
    session.add(pacient)
    session.commit()
    session.refresh(pacient)
    return pacient