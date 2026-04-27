
from typing import TypedDict, List, Literal, Optional
from pydantic import BaseModel, Field


N_SEARCH_RESULTS=4
MAX_ITERATIONS=5
TARGET_ARTICLES=3
MIN_SCORE=7.0
LIMIT_SLEEP=1.5

class MedicalResearchState(TypedDict):
    #Inputs
    patient_data: dict
    search_focus: str
    #images: Optional[List[str]] | None= None
    
    #Node 1: Interpretation
    clinical_summary: str
    queries: List[str]
    
    #Node 2: Search and evaluation
    current_query_index: int
    raw_articles: List[dict]
    urls_seen: List[str]
    articles_punctuation: List[dict]
    iterations: int
    
    #Node 5: Selection
    best_articles: List[dict]
    
    # Node 6: Final analysis 
    final_reports: List[dict]
    
    #Control
    search_completion: bool

class ClinicalInterpretation(BaseModel):
    main_diagnosis: str =Field(description="Main diagnosis identified")
    key_findings: List[str]=Field(description="Clinical findings most relevant (up to 6)")
    comorbidity: List[str]= Field(description="Relevant comorbidities")
    clinical_questions: List[str] = Field(description="Clinical Questions the bibliography must answer (up to 5)")
    
    queries_pubmed: List[str] =Field(
        description=
        "5 optimized queries for PubMed, EVERYTHING in line with the"
        "bibliographical focus presented in the system prompt."
        "The focus inside the prompt may vary"
        "(don't repeat the same question)"
        "Treatment example: 'heart failure reduced ejection fraction SGLT2 inhibitors RCT'"
    )
    
class ArticleScore(BaseModel):
    punctuation: float=Field(ge=0, le=10, description="Relevance 0-10")
    evidence_level:str =Field (description="Type: "
                                "meta-analysis, RCT, cohort study, Case-control study, systematic review, narrative review, case report, clinical guideline,other")
    is_relevant: bool = Field(description= f"True if punctuation  >= {MIN_SCORE}")
    justification: str =Field(description= "Brief reason (1-2 sentences)")

class ArticleAnalysis(BaseModel):
    title: str
    source_url: str
    abstract: str
    clinical_analysis: str =Field(description="Specific association with the patient, diagnostic or therapeutic implications, in ~150 words")
    evidence_level:str
    relevance_punctuation: float

focus_instructions={
    "epidemiology": (
"The focus is EPIDEMIOLOGICAL. Generate queries oriented toward: incidence, prevalence, "
"risk factors, morbidity/mortality, burden of disease, risk groups "
"and population trends. Avoid treatment or diagnostic technique articles."
),
"diagnosis": (
"The focus is DIAGNOSTIC. Generate queries oriented toward: diagnostic criteria, "
"sensitivity/specificity of tests, biomarkers, imaging, differential diagnosis, "
"clinical utility of diagnostic tools and evaluation guidelines. "
"Avoid treatment or epidemiology articles."
),
"treatment": (
"The focus is THERAPEUTIC. Generate queries oriented toward: pharmacological interventions, "
"procedures, therapeutic goals, comparison of regimens, RCTs, management guidelines, "
"relevant adverse effects and adherence. Avoid diagnostic or epidemiology articles."
),
}