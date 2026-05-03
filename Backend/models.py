import datetime
from typing import Optional
from sqlmodel import Field, SQLModel
from pydantic import BaseModel

class PacientBase(SQLModel):
    pacient_name: Optional[str]=Field(default=None)
    sex: str
    allergies:Optional[str]=Field(default=None)
    relevant_history:Optional[str]=Field(default=None)
    previous_hospitalizations:Optional[str]=Field(default=None)
    
    update_date: Optional[str]=Field(default_factory=lambda: datetime.today().isoformat())

class PacientCreate(PacientBase):
    pass

class PacientPublic(PacientBase):
    id: int

class PacientUpdate(SQLModel):
    pacient_name: str| None=None
    sex: str| None=None
    allergies: str| None=None
    relevant_history: str| None=None
    previous_hospitalizations: str| None=None
    update_date: str| None=None

class PacientList(BaseModel):
    items: list[PacientPublic]
    total: int
    offset: int
    limit: int

class PacientDB(PacientBase, Table=True):
    __tablename__="pacient"
    id: Optional[int]=Field(default=None, primary_key=True)
    
class ConsultBase(SQLModel):
    pacient_id: int
    
    main_complain: str
    primary_diagnosis: Optional[str]=Field(default=None)
    secondary_diagnosis: Optional[str]=Field(default=None)
    symptoms: list[str]
    vitals: list[str]
    exams_results: Optional[str]=Field(default=None)
    current_medication: Optional[list[str]]=Field(default=None)
    
    update_date: Optional[str]=Field(default_factory=lambda: datetime.today().isoformat())
    
class ConsultCreate(ConsultBase):
    pass

class ConsultPublic(ConsultBase):
    id: int
    
class ConsultUpdate(SQLModel):
    pacient_id: int| None=None
    
    main_complain: str| None=None
    primary_diagnosis: str| None=None
    secondary_diagnosis: str| None=None
    symptoms: list[str]| None=None
    vitals: list[str]| None=None
    exams_results: str| None=None
    current_medication: list[str]| None=None
    
    update_date: str| None=None

class ConsultList(BaseModel):
    items: list[ConsultPublic]
    total: int
    offset: int
    limit: int
    
class ConsultDB(ConsultBase, Table=True):
    __tablename__="consult"
    id: Optional[int]=Field(default=None, primary_key=True)
    pacient_id: int=Field(foreign_key="pacient.id")
    
    
class AnalysisBase(SQLModel):
    consult_id: int
    
    tile: str
    source: str
    abstract: str
    clinical_analysis: str
    evidence_level: str
    relevance_punctuation: float

class AnalysisCreate(AnalysisBase):
    pass

class AnalysisPublic(AnalysisBase):
    id: int

class AnalysisUpdate(SQLModel):
    consult_id: int| None=None
    
    tile: str| None=None
    source: str| None=None
    abstract: str| None=None
    clinical_analysis: str| None=None
    evidence_level: str| None=None
    relevance_punctuation: float| None=None

class AnalysisList(BaseModel):
    items: list[AnalysisPublic]
    total: int
    offset: int
    limit: int

class AnalysisDB(AnalysisBase, Table=True):
    __tablename__="analysis"
    id: Optional[int]=Field(default=None, primary_key=True)
    consult_id: int=Field(foreign_key="consult.id")
    
