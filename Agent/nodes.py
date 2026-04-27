from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import PubMedAPIWrapper
import json
import os
from dotenv import load_dotenv
from schema import (MedicalResearchState, ClinicalInterpretation, 
                    focus_instructions, N_SEARCH_RESULTS, MAX_ITERATIONS,
                    MIN_SCORE, TARGET_ARTICLES, LIMIT_SLEEP)

# LLM and search engine configuration
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=api_key)

ddg_wrapper=DuckDuckGoSearchAPIWrapper(max_results=N_SEARCH_RESULTS, time="y")
ddg_search=DuckDuckGoSearchResults(api_wrapper=ddg_wrapper, output_format="list")

# Nodes

# Node 1: analyze pacient
def analyze_pacient(state: MedicalResearchState)-> dict:
    focus=state['search_focus']
    instruction=focus_instructions[focus]
    
    patient_json=json.dumps(state["patient_data"], ensure_ascii=False, indent=2)
    llm_structured=LLM.with_structured_output(ClinicalInterpretation)
    
    response: ClinicalInterpretation =llm_structured.invoke([
        SystemMessage(Content=(
            "You are an internist with a subspecialty in evidence-based medicine. "
    "Analyze the clinical history and generate search queries for PubMed.\n\n"
    f"MANDATORY FOCUS: {instruction}\n\n"
    "All queries must answer questions within that specific focus. "
    "Do not mix different focuses other than the one indicated."
        )),
        HumanMessage(content=f"Historial clínico: \n {patient_json}")
    ])
    
    summary=(
        f"Diagnosis: {response.main_diagnosis}\n"
        f"Key findings: {", ".join(response.key_findings)}\n"
        f"Comorbidities: {", ".join(response.comorbidity)}\n"
        f"Clinical questions: {", ".join(response.clinical_questions)}"
    )
    
    return{
        "clinical_summary": summary,
        "queries":response.queries_pubmed,
        "current_query_index":0,
        "raw_articles":[],
        "urls_seen": [],
        "articles_punctuation": [],
        "iterations": 0,
        "search_completion": False,
    }
    
#Node 2: Search articles, try pubmed API then duckduckgo
def articles_search(state: MedicalResearchState)-> dict:
        
    index=state["current_query_index"]
    queries=state["queries"]
    
    if index>= len(queries):
        return {"search_completion":True}
    
    query=queries[index]
    seen=set(state.get("urls_seen",[]))
    new=[]
    new_url=[]
    
    try:
        pubmed_results=pubmed.run(query)
        
    medical_sites=("site:pubmed.ncbi.nlm.nih.gov OR site:nejm.org OR site:thelancet.com "
        "OR site:jamanetwork.com OR site:bmj.com OR site:ahajournals.org "
        "OR site:nature.com OR site:uptodate.com")
    consult_query=f"{query}({medical_sites})"
    try:
        time.sleep(LIMIT_SLEEP)
        results=consult_query.invoke(consult_query)
        if isinstance(results, str):
            results=json.loads(results)