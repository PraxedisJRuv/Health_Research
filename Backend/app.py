from fastapi import FastAPI, Depends, HTTPException, Response, Query
from sqlmodel import Session, select, or_, func
from typing import Literal
from .models import *
from .utils import get_session, create_db
from Agent.graph import graph_build

app=FastAPI()

create_db()

graph = graph_build()



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

@app.post("/pacients/{pacient_id}/consults", response_model=ConsultPublic)
def create_consult(pacient_id: int, consult: ConsultCreate, session=Depends(get_session)):
    consult.pacient_id = pacient_id
    db_consult = ConsultDB.model_validate(consult)
    session.add(db_consult)
    session.commit()
    session.refresh(db_consult)
    return db_consult

@app.post("/consults", response_model=ConsultPublic)
def create_consult(consult: ConsultCreate, session=Depends(get_session)):
    db_consult=ConsultDB.model_validate(consult)
    session.add(db_consult)
    session.commit()
    session.refresh(db_consult)
    return db_consult

@app.get("/consults", response_model=ConsultList)
def get_consults_list(offset: int=0, limit: int=Query(default=10, le=100), search: str|None=None, session=Depends(get_session)):
    query=select(ConsultDB)
    if search:
        query=query.where(or_(
            ConsultDB.main_complain.ilike(f"%{search}%"),
            ConsultDB.primary_diagnosis.ilike(f"%{search}%"),
            ConsultDB.update_date.ilike(f"%{search}%")
        ))
    total=session.exec(select(func.count()).select_from(query.subquery())).one()
    consults=session.exec(query.offset(offset).limit(limit)).all()
    return ConsultList(items=consults, total=total, offset=offset, limit=limit)

@app.get("/consults/{consult_id}", response_model=ConsultPublic)
def get_consult(consult_id: int, session=Depends(get_session)):
    consult=session.get(ConsultDB, consult_id)
    if not consult:
        raise HTTPException(status_code=404, detail="Consult not found")
    return consult

@app.get("/pacients/{pacient_id}/consults", response_model=ConsultList)
def get_consults_by_pacient(pacient_id: int, offset: int = 0, limit: int = Query(default=10, le=100), session=Depends(get_session)):
    query = select(ConsultDB).where(ConsultDB.pacient_id == pacient_id)
    
    total = session.exec(select(func.count()).select_from(query.subquery())).one()
    consults = session.exec(query.offset(offset).limit(limit)).all()
    
    return ConsultList(items=consults, total=total, offset=offset, limit=limit)

@app.delete("/consults/{consult_id}")
def delete_consult(consult_id: int, session=Depends(get_session)):
    consult=session.get(ConsultDB, consult_id)
    if not consult:
        raise HTTPException(status_code=404, detail="Consult not found")
    session.delete(consult)
    session.commit()
    return Response(status_code=204)

@app.patch("/consults/{consult_id}", response_model=ConsultPublic)
def update_consult(consult_id: int, update: ConsultUpdate, session=Depends(get_session)):
    consult=session.get(ConsultDB, consult_id)
    if not consult:
        raise HTTPException(status_code=404, detail="Consult not found")
    
    update_data=update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(consult, key, value)
    
    session.add(consult)
    session.commit()
    session.refresh(consult)
    return consult

@app.get("/consults/{consult_id}/analysis", response_model=AnalysisList)
def get_analysis_by_consult(consult_id: int, offset: int = 0, limit: int = Query(default=10, le=100), session=Depends(get_session)):
    query = select(AnalysisDB).where(AnalysisDB.consult_id == consult_id)
    
    total = session.exec(select(func.count()).select_from(query.subquery())).one()
    analysis = session.exec(query.offset(offset).limit(limit)).all()
    
    return AnalysisList(items=analysis, total=total, offset=offset, limit=limit)

@app.post("/analyses", response_model=AnalysisList)
def create_analyses(request: AnalysisRequest, session=Depends(get_session)):
    consult_id = request.consult_id
    focus = request.focus
    
    consult = session.get(ConsultDB, consult_id)
    if not consult:
        raise HTTPException(status_code=404, detail="Consult not found")
    
    pacient = session.get(PacientDB, consult.pacient_id)
    if not pacient:
        raise HTTPException(status_code=404, detail="Pacient not found")
    
    patient_data = {
        "pacient": pacient.model_dump(),
        "consult": consult.model_dump()
    }
    
    initial_state = {
        "patient_data": patient_data,
        "search_focus": focus
    }
    
    result = graph.invoke(initial_state)
    final_report = result.get("final_report", [])
    
    analyses = []
    for article in final_report:
        analysis = AnalysisDB(
            consult_id=consult_id,
            pacient_id=pacient.id,
            **article
        )
        session.add(analysis)
        analyses.append(analysis)
    
    session.commit()
    for a in analyses:
        session.refresh(a)
    
    return AnalysisList(items=analyses, total=len(analyses), offset=0, limit=len(analyses))