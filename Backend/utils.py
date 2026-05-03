import json
import os
from sqlmodel import Session, SQLModel, create_engine

sqlite_file_name="database.db"
sqlite_url=f"sqlite:///{sqlite_file_name}"

connect_args={"check_same_thread":False}
engine=create_engine(url=sqlite_url, connect_args=connect_args)

def create_db():
    SQLModel.metadata.create_all(engine)
    
def get_session():
    with Session(engine) as session:
        yield session