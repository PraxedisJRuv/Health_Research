
from typing import Literal
from .schema import (MedicalResearchState, 
                    MIN_SCORE, TARGET_ARTICLES, MAX_ITERATIONS)
def stop_criteria(state: MedicalResearchState)-> Literal["article_search", "select_top"]:
    high_quality=[
        a for a in state.get("articles_punctuation", [])
        if a.get("punctuation", 0)>= MIN_SCORE
    ]
    
    if len(high_quality)>= TARGET_ARTICLES:
        print("Enough quality articles")
        return "select_top"
    
    if state["current_query_index"]>= len(state["queries"]):
        print("queries depleted")
        return "select_top"
    
    if state["iterations"]>= MAX_ITERATIONS:
        print(f"Reached maximum iterations ({MAX_ITERATIONS})")
        return "select_top"
    
    return "search_articles"