from langgraph.graph import StateGraph, END
from .schema import MedicalResearchState

from .edges import stop_criteria
from .nodes import (analyze_pacient, articles_search, punctuate_articles,
                    select_top, article_analysis,fetch_and_chunk,semantic_retrieval)#, print_report)

def graph_build() -> StateGraph:
    #initialize nodes and state
    workflow=StateGraph(MedicalResearchState)
    workflow.add_node("analyze_pacient", analyze_pacient)
    workflow.add_node("articles_search",articles_search)
    workflow.add_node("punctuate_articles", punctuate_articles)
    workflow.add_node("select_top",select_top)
    workflow.add_node("article_analysis",article_analysis)
    #workflow.add_node("print_report", print_report)
    
    #RAG nodes
    workflow.add_node("fetch_and_chunk", fetch_and_chunk)
    workflow.add_node("semantic_retrieval",semantic_retrieval)
    
    #initialize graph
    workflow.set_entry_point("analyze_pacient")
    workflow.add_edge("analyze_pacient","articles_search")
    workflow.add_edge("articles_search", "punctuate_articles")
    workflow.add_edge("articles_search","fetch_and_chunk")
    
    workflow.add_conditional_edges("punctuate_articles", 
                                    stop_criteria, 
                                    {
                                        "articles_search":"articles_search",
                                        "select_top":"select_top",
                                    },
                                )
    workflow.add_edge("select_top","semantic_retrieval")
    workflow.add_edge("semantic_retrieval","article_analysis")
    #workflow.add_edge("article_analysis", "print_report")
    #workflow.add_edge("print_report",END)
    workflow.add_edge("article_analysis", END)
    
    return workflow.compile()

"""
    While the RAG version is techincally much better, and accuarate to the intend of the system
    it is also true that it might end un being much more expensive. Due to the analysis including the embbedings, 
    making this wouldn't represent a big increase.
    
    it is supposed to be around 800 tokens for the pacient analyisis
    400 for every article punctuation so 400*N
    
    not really sure for the emmbeding, but it seems around 500 in total
    
    the analyisis without RAG would be 1200 and with RAG iw would be at least tripled, but up to 6000.
"""