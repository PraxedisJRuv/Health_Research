from schema import MedicalResearchState
from graph import graph_build

EXAMPLE={"patient_id": "MX-2024-001",
    "age": 67,
    "sex": "Male",
    "chief_complaint": "Progressive dyspnea over 3 months, lower extremity edema",
    "primary_diagnosis": "Heart failure with reduced ejection fraction (HFrEF)",
    "secondary_diagnoses": [
        "Type 2 diabetes mellitus",
        "Systemic arterial hypertension",
        "Paroxysmal atrial fibrillation",
    ],
    "medical_history": {
        "symptoms": ["exertional dyspnea", "orthopnea", "paroxysmal nocturnal dyspnea", "bilateral edema"],
        "vital_signs": {"BP": "145/90 mmHg", "HR": "88 bpm", "RR": "22 rpm", "SpO2": "94%"},
        "lab_results": {
            "BNP":         "850 pg/mL",
            "creatinine":  "1.4 mg/dL",
            "HbA1c":       "8.2%",
            "sodium":      "134 mEq/L",
            "potassium":   "4.1 mEq/L",
        },
        "echocardiogram": "LVEF 30%, LV dilation, moderate mitral regurgitation",
        "ECG": "Atrial fibrillation, left bundle branch block (LBBB), QRS 140ms",
    },
    "current_medications": [
        "Metformin 850 mg q12h",
        "Enalapril 10 mg q12h",
        "Furosemide 40 mg q24h",
        "Bisoprolol 5 mg q24h",
        "Warfarin 5 mg q24h (target INR 2-3)",
    ],
    "allergies": "NSAIDs (urticaria)",
    "relevant_history": "Anterior acute myocardial infarction 2 years ago. Catheterization: LAD with 60% residual lesion.",
    "previous_hospitalizations": 2,
    }
FOCUS="treatment"

def run():
    
    graph=graph_build()
    
    initial_state:MedicalResearchState={
    "patient_data": EXAMPLE,
    "search_focus": FOCUS,    
    "clinical_summary":"",
    "queries":[],
    "current_query_index":0,
    "raw_articles":[],
    "urls_seen":[],
    "articles_punctuation":[],
    "iterations": 0,
    "best_articles":[],
    "final_report":[],
    "search_completion":False,
    }
    
    result=graph.invoke(initial_state)


    return result

run()