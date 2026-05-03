from langgraph.graph import StateGraph, END
from .schema import MedicalResearchState

from .edges import stop_criteria
from .nodes import (analyze_pacient, articles_search, punctuate_articles,
                   select_top, article_analysis)#, print_report)

def graph_build() -> StateGraph:
    workflow=StateGraph(MedicalResearchState)
    workflow.add_node("analyze_pacient", analyze_pacient)
    workflow.add_node("articles_search",articles_search)
    workflow.add_node("punctuate_articles", punctuate_articles)
    workflow.add_node("select_top",select_top)
    workflow.add_node("article_analysis",article_analysis)
    #workflow.add_node("print_report", print_report)
    
    workflow.set_entry_point("analyze_pacient")
    
    workflow.add_edge("analyze_pacient","articles_search")
    workflow.add_edge("articles_search", "punctuate_articles")
    
    workflow.add_conditional_edges("punctuate_articles", 
                                   stop_criteria, 
                                   {
                                       "articles_search":"articles_search",
                                       "select_top":"select_top",
                                   },
                                )
    workflow.add_edge("select_top","article_analysis")
    #workflow.add_edge("article_analysis", "print_report")
    #workflow.add_edge("print_report",END)
    workflow.add_edge("article_analysis", END)
    
    return workflow.compile()