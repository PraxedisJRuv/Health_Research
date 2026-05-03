from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import PubMedAPIWrapper
import json
import os
from dotenv import load_dotenv
from Agent.schema import (MedicalResearchState, ClinicalInterpretation, ArticleScore,
                    ArticleAnalysis,focus_instructions, N_SEARCH_RESULTS, 
                    MAX_ITERATIONS, MIN_SCORE, TARGET_ARTICLES, LIMIT_SLEEP)

# LLM and search engine configuration
load_dotenv()
api_key=os.getenv("GOOGLE_API_KEY")
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0, api_key=api_key)

ddg_wrapper=DuckDuckGoSearchAPIWrapper(max_results=N_SEARCH_RESULTS, time="y")
ddg_search=DuckDuckGoSearchResults(api_wrapper=ddg_wrapper, output_format="list")

pubmed = PubMedAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=2000,
    email=os.getenv("NCBI_EMAIL", "investigacion@hospital.mx"),
)

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
        if pubmed_results and "Title:" in pubmed_results:
            bloques = pubmed_results.split("\n\n")
            for bloque in bloques:
                if "Title:" not in bloque:
                    continue
                lines=bloque.strip().split("\n")
                title=next((l.replace("Title:", "").strip() for l in lines if l.startswith("Title:")), "")
                uid    = next((l.replace("Published:", "").strip() for l in lines if l.startswith("Published:")), "")
                snippet= "\n".join(l for l in lines if not l.startswith(("Title:", "Published:", "Copyright")))
                url    = f"https://pubmed.ncbi.nlm.nih.gov/?term={'+'.join(title.split()[:5])}"

                if title and url not in seen:
                    seen.add(url)
                    new.append({"title": title, "snippet": snippet[:800], "url":url, "source":"PubMed","query_used":query})
                    new_url.append(url)
    except Exception as error:
        print(f" PubMed error: {error}")

    medical_sites=("site:pubmed.ncbi.nlm.nih.gov OR site:nejm.org OR site:thelancet.com "
        "OR site:jamanetwork.com OR site:bmj.com OR site:ahajournals.org "
        "OR site:nature.com OR site:uptodate.com")
    consult_query=f"{query}({medical_sites})"
    try:
        time.sleep(LIMIT_SLEEP)
        results=consult_query.invoke(consult_query)
        if isinstance(results, str):
            results=json.loads(results)
            
        before=len(new)
        for r in results:
            url=r.get("link",r.get("url", ""))
            title=r.get("title", "")
            if url and url not in seen:
                seen.add(url)
                new.append({"title": title, "snippet": r.get("snippet",r.get("body", ""))[:800], "url":url, "source":"DuckDuckGo","query_used":query})
            new_url.append(url)
            
    except Exception as error:
        print(f"DuckDuckGo error: {error}")
    
    
    return {
        "raw articles": state["raw_articles"] + new,
        "urls_seen": state.get("urls_seen", []) +new_url,
        "current_query_index": index +1,
        "iterations": state["iterations"]+1,
    }


# Node 3: Evaluate relevance
def punctuate_articles(state: MedicalResearchState)-> dict:
    punctuated={a["url"] for a in state.get("articles_punctuation", [])}
    to_punctuate=[a for a in state["raw_articles"] if a ["url"] not in punctuated]

    if not to_punctuate:
        return{}
    
    llm_scorer=LLM.with_structured_output(ArticleScore)
    now_punctuated=[]
    
    for article in to_punctuate:
        try:
            score:ArticleScore=llm_scorer.invoke([
                SystemMessage(content=(
                    "You are an expert in evidence-based medicine. "
                    "Evaluate the relevance of this article to the clinical case. "
                    "Consider diagnostic and therapeutic relevance, as well as the level of evidence. "
                    f"Relevant = score >= {MIN_SCORE}."
                )),
                HumanMessage(content=(
                    f"CLINICAL CONTEXT:\n{state["clinical_summary"]}\n\n"
                    f"ARTICLE:\nTitle: {article['title']}\n"
                    f"Snippet: {article['snippet']}\n"
                    f"Source: {article['url']}"
                ))
            ])
            now_punctuated.append({
                **article,
                "punctuation": score.punctuation,
                "evidence_level": score.evidence_level,
                "is relevant": score.is_relevant,
                "justification": score.justification,
            })
            
            #relevance="True" if score.is_relevant else "False"
            #print(f"   {relevance} [{score.punctuation:.1f}/10] [{score.evidence_level}] {article['title'][:70]}...")

        except Exception as error:
            print(f"Error ranking '{article["title"]}':{error}")
            continue
        
    total_punctuated = sorted(
        state.get("articles_punctuation", [])+now_punctuated,
        key=lambda x:x.get("punctuation",0),
        reverse=True,
    )

    return {"articles_punctuated": total_punctuated}


# Node 5: Select top
def select_top(state: MedicalResearchState) -> dict:
    punctuated=state.get("articles_punctuation",[])
    if not punctuated:
        print("Any articles were found")
        return{"best_articles": []}
    
    high_quality=[a for a in punctuated if a.get("punctuation", 0)>= MIN_SCORE]
    others=[a for a in punctuated if a.get("punctuation", 0)< MIN_SCORE]
    candidates=(high_quality+others)[:TARGET_ARTICLES]
    
    #for i,a in enumerate(candidates, 1):
        #print(f"    #{i} [{a.get("punctuation", 0):.1f}/10]")
        
    return {"best_articles":candidates}

def article_analysis(state: MedicalResearchState)-> dict:
    llm_analyst=LLM.with_structured_output(ArticleAnalysis)
    patient_json=json.dump(state["patient_data"], ensure_ascii=False, indent=2)
    report=[]
    
    for i, article in enumerate(state["best_articles"], 1):
        #print(f"    Analyzing #{i}: {article["title"]}...")
        try:
            analysis: ArticleAnalysis=llm_analyst.invoke([
                SystemMessage(content=(
                    "You are a physician drafting a literature review report. "
                    "Write with clinical rigor. "
                    "The abstract summary must be faithful to the article's content. "
                    "The clinical analysis must be specific to THIS patient, "
                    "mentioning specific diagnostic or therapeutic implications."
                )),
                HumanMessage(content=(
                    f"FULL PATIENT HISTORY:\n{patient_json}\n\n"
                    f"CLINICAL SUMMARY:\n{state["clinical_summary"]}\n\n"
                    f"ARTICLE #{i}:\n"
                    f"Title: {article['title']}\n"
                    f"URL/Source: {article['url']}\n"
                    f"Available content: {article['snippet']}\n"
                    f"Level of evidence: {article.get('evidence_level', 'Not determined')}\n"
                    f"Relevance score: {article.get('punctuation', 'N/A')}/10\n"
                    f"Evaluator justification: {article.get('justification', '')}"
                ))
            ])
            report.append(analysis.model_dump())
        except Exception as error:
            print(f"    Error analyzing article #{i}: {error}")
            report.append({
                "title": article.get("title", "without title"),
                "source_url": article.get("url", ""),
                "abstract": article.get("snippet", "Not available"),
                "clinical_analysis": "It wasn't possible to generate an analysis",
                "evidence_level": article.get("evidence_level", "unknown"),
                "relevance_punctuation": article.get("punctuation",0),
            })
    return{"final_report":report}

#def print_report(state: MedicalResearchState) -> dict:
    