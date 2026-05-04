import datetime
from typing import Optional, Literal, List
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON
from pydantic import BaseModel

class PacientBase(SQLModel):
    pacient_name: Optional[str]=Field(default=None)
    sex: str
    allergies:Optional[str]=Field(default=None)
    relevant_history:Optional[str]=Field(default=None)
    previous_hospitalizations:Optional[int]=Field(default=None)
    
    update_date: Optional[str]=Field(default_factory=lambda: datetime.today().isoformat())
#Hay que revisar el datetime today
class PacientCreate(PacientBase):
    pass

class PacientPublic(PacientBase):
    id: int

class PacientUpdate(SQLModel):
    pacient_name: str| None=None
    sex: str| None=None
    allergies: str| None=None
    relevant_history: str| None=None
    previous_hospitalizations: int| None=None
    update_date: str| None=None

class PacientList(BaseModel):
    items: List[PacientPublic]
    total: int
    offset: int
    limit: int

class PacientDB(PacientBase, table=True):
    __tablename__="pacient"
    id: Optional[int]=Field(default=None, primary_key=True)
    
class ConsultBase(SQLModel):
    pacient_id: int| None=None
    
    main_complain: str
    primary_diagnosis: Optional[str]=Field(default=None)
    secondary_diagnosis: Optional[str]=Field(default=None)
    symptoms: List[str]=Field(sa_column=Column(JSON))
    vitals: List[str]=Field(sa_column=Column(JSON))
    exams_results: Optional[str]=Field(default=None)
    current_medication: Optional[List[str]]=Field(default=None, sa_column=Column(JSON))
    
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
    symptoms: List[str]| None=None
    vitals: List[str]| None=None
    exams_results: str| None=None
    current_medication: List[str]| None=None
    
    update_date: str| None=None

class ConsultList(BaseModel):
    items: List[ConsultPublic]
    total: int
    offset: int
    limit: int
    
class ConsultDB(ConsultBase, table=True):
    __tablename__="consult"
    id: Optional[int]=Field(default=None, primary_key=True)
    pacient_id: int=Field(foreign_key="pacient.id")
    
    
class AnalysisBase(SQLModel):
    consult_id: int
    
    focus:str
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
    items: List[AnalysisPublic]
    total: int
    offset: int
    limit: int

class AnalysisRequest(BaseModel):
    consult_id: int
    focus: Literal["epidemiology", "diagnosis", "treatment"]

class AnalysisDB(AnalysisBase, table=True):
    __tablename__="analysis"
    id: Optional[int]=Field(default=None, primary_key=True)
    consult_id: int=Field(foreign_key="consult.id")
    pacient_id: int=Field(foreign_key="pacient.id")
    
