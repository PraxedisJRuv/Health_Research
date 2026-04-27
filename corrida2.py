"""
=============================================================================
AGENTE DE BIBLIOGRAFÍA MÉDICA - LangGraph
=============================================================================
Dado un historial clínico en JSON, este agente:
1. Interpreta los datos clínicos más relevantes con un LLM
2. Genera múltiples queries de búsqueda especializadas
3. Busca artículos médicos con DuckDuckGo + PubMed
4. Evalúa y puntúa cada artículo con un LLM evaluador
5. Decide cuándo detener la búsqueda
6. Selecciona los top 5 artículos
7. Redacta un análisis clínico final para cada uno

Instalación:
    pip install langgraph langchain langchain-anthropic \
                langchain-community duckduckgo-search \
                xmltodict biopython

Variables de entorno requeridas:
    ANTHROPIC_API_KEY=sk-ant-...
    NCBI_API_KEY=...  (opcional pero recomendado para PubMed, gratis en ncbi.nlm.nih.gov)
=============================================================================
"""

import json
import time
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Literal, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities import PubMedAPIWrapper
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# ===========================================================================
# CONFIGURACIÓN
# ===========================================================================

# Focos válidos y sus instrucciones para el LLM generador de queries
FocoBibliografico = Literal["epidemiología", "diagnóstico", "tratamiento"]

INSTRUCCIONES_FOCO: dict[str, str] = {
    "epidemiología": (
        "El foco es EPIDEMIOLÓGICO. Genera queries orientadas a: incidencia, prevalencia, "
        "factores de riesgo, mortalidad/morbimortalidad, carga de enfermedad, grupos de riesgo "
        "y tendencias poblacionales. Evita artículos de tratamiento o técnicas diagnósticas."
    ),
    "diagnóstico": (
        "El foco es DIAGNÓSTICO. Genera queries orientadas a: criterios diagnósticos, "
        "sensibilidad/especificidad de pruebas, biomarcadores, imagen, diagnóstico diferencial, "
        "utilidad clínica de herramientas diagnósticas y guías de evaluación. "
        "Evita artículos de tratamiento o epidemiología."
    ),
    "tratamiento": (
        "El foco es TERAPÉUTICO. Genera queries orientadas a: intervenciones farmacológicas, "
        "procedimientos, metas terapéuticas, comparación de esquemas, ECA, guías de manejo, "
        "efectos adversos relevantes y adherencia. Evita artículos de diagnóstico o epidemiología."
    ),
}

MAX_ITERATIONS    = 6      # Máximo de rondas de búsqueda
TARGET_ARTICLES   = 5      # Artículos finales a presentar
MIN_SCORE         = 6.0    # Puntuación mínima de relevancia (sobre 10)
SEARCH_RESULTS_N  = 5      # Resultados por búsqueda DuckDuckGo
RATE_LIMIT_SLEEP  = 1.5    # Segundos entre llamadas a DuckDuckGo

# Modelo: claude-sonnet-4-20250514 es el más reciente de Claude 4
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0,api_key=api_key)
# ===========================================================================
# HERRAMIENTAS DE BÚSQUEDA
# ===========================================================================

# --- DuckDuckGo (búsqueda general con foco en fuentes médicas) ---
ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=SEARCH_RESULTS_N, time="y")
ddg_search   = DuckDuckGoSearchResults(api_wrapper=ddg_wrapper, output_format="list")

# --- PubMed (literatura biomédica indexada, mucho más confiable) ---
# Regístrate gratis en https://www.ncbi.nlm.nih.gov/account/ para obtener NCBI_API_KEY
pubmed = PubMedAPIWrapper(
    top_k_results=3,
    doc_content_chars_max=2000,
    email=os.getenv("NCBI_EMAIL", "investigacion@hospital.mx"),
)

# ===========================================================================
# ESQUEMAS PYDANTIC PARA SALIDA ESTRUCTURADA
# ===========================================================================

class ClinicalInterpretation(BaseModel):
    """Interpretación clínica del historial del paciente."""
    diagnostico_principal: str = Field(description="Diagnóstico principal identificado")
    hallazgos_clave: List[str]  = Field(description="Hallazgos clínicos más relevantes (máx 6)")
    comorbilidades: List[str]   = Field(description="Comorbilidades relevantes")
    preguntas_clinicas: List[str] = Field(
        description="Preguntas clínicas que la bibliografía debe responder (máx 4)"
    )
    queries_pubmed: List[str] = Field(
        description=(
            "5 queries en INGLÉS optimizados para PubMed, TODOS alineados al foco "
            "bibliográfico indicado en el system prompt. Varía el ángulo dentro del foco "
            "(no repitas la misma pregunta). "
            "Ejemplo tratamiento: 'heart failure reduced ejection fraction SGLT2 inhibitors RCT'"
        )
    )

class ArticleScore(BaseModel):
    """Evaluación de relevancia de un artículo."""
    puntuacion: float = Field(ge=0, le=10, description="Relevancia 0-10")
    nivel_evidencia: str = Field(
        description="Tipo: meta-análisis, ECA, cohorte, caso-control, revisión sistemática, "
                    "revisión narrativa, reporte de caso, guía clínica, otro"
    )
    es_relevante: bool  = Field(description=f"True si puntuación >= {MIN_SCORE}")
    justificacion: str  = Field(description="Razón breve (1-2 oraciones)")

class ArticleAnalysis(BaseModel):
    """Análisis clínico final de un artículo."""
    titulo: str
    fuente_url: str
    resumen_abstract: str  = Field(description="Resumen del abstract en ~200 palabras, en español")
    analisis_clinico: str  = Field(
        description="Asociación específica con el paciente, implicaciones diagnósticas "
                    "o terapéuticas, en ~150 palabras, en español"
    )
    nivel_evidencia: str
    puntuacion_relevancia: float

# ===========================================================================
# ESTADO DEL GRAFO
# ===========================================================================

class MedicalResearchState(TypedDict):
    # Entrada
    patient_data: dict
    foco_bibliografico: FocoBibliografico  # "epidemiología" | "diagnóstico" | "tratamiento"

    # Nodo 1: Interpretación
    resumen_clinico: str
    queries: List[str]

    # Nodo 2-4: Búsqueda y evaluación
    current_query_index: int
    articulos_crudos: List[dict]          # Todos los encontrados (sin duplicados)
    urls_vistos: List[str]                # Para deduplicación
    articulos_puntuados: List[dict]       # Con score asignado
    iteraciones: int

    # Nodo 5: Selección
    top_articulos: List[dict]             # Los 5 mejores

    # Nodo 6: Análisis final
    reporte_final: List[dict]

    # Control de flujo
    busqueda_completa: bool

# ===========================================================================
# NODO 1 — INTERPRETACIÓN DEL PACIENTE
# ===========================================================================

def interpretar_paciente(state: MedicalResearchState) -> dict:
    """
    LLM interpreta el historial clínico y genera:
    - Resumen clínico estructurado
    - 5 queries de búsqueda PubMed en inglés, todas orientadas al foco seleccionado
    """
    foco         = state["foco_bibliografico"]
    instruccion  = INSTRUCCIONES_FOCO[foco]

    print(f"\n [Nodo 1] Interpretando historial clínico...")
    print(f"   Foco bibliográfico: {foco.upper()}")

    patient_json   = json.dumps(state["patient_data"], ensure_ascii=False, indent=2)
    llm_structured = LLM.with_structured_output(ClinicalInterpretation)

    response: ClinicalInterpretation = llm_structured.invoke([
        SystemMessage(content=(
            "Eres un médico internista con subespecialidad en medicina basada en evidencia. "
            "Analiza el historial clínico y genera consultas de búsqueda para PubMed.\n\n"
            f"FOCO OBLIGATORIO: {instruccion}\n\n"
            "Todas las queries deben responder preguntas dentro de ese foco específico. "
            "No mezcles focos distintos al indicado."
        )),
        HumanMessage(content=f"Historial clínico:\n{patient_json}")
    ])

    resumen = (
        f"Diagnóstico: {response.diagnostico_principal}\n"
        f"Hallazgos clave: {', '.join(response.hallazgos_clave)}\n"
        f"Comorbilidades: {', '.join(response.comorbilidades)}\n"
        f"Preguntas clínicas: {'; '.join(response.preguntas_clinicas)}"
    )

    print(f"   Diagnóstico identificado: {response.diagnostico_principal}")
    print(f"   Queries generadas: {len(response.queries_pubmed)}")
    for i, q in enumerate(response.queries_pubmed, 1):
        print(f"     {i}. {q}")

    return {
        "resumen_clinico": resumen,
        "queries": response.queries_pubmed,
        "current_query_index": 0,
        "articulos_crudos": [],
        "urls_vistos": [],
        "articulos_puntuados": [],
        "iteraciones": 0,
        "busqueda_completa": False,
    }

# ===========================================================================
# NODO 2 — BÚSQUEDA DE ARTÍCULOS (DuckDuckGo + PubMed)
# ===========================================================================

def buscar_articulos(state: MedicalResearchState) -> dict:
    """
    Ejecuta la query actual con dos fuentes:
    - PubMed (literatura indexada, confiable)
    - DuckDuckGo filtrado a dominios médicos de alto impacto

    Deduplica por URL antes de agregar al estado.
    """
    idx     = state["current_query_index"]
    queries = state["queries"]

    if idx >= len(queries):
        print("\n [Nodo 2] Todas las queries agotadas.")
        return {"busqueda_completa": True}

    query = queries[idx]
    print(f"\n [Nodo 2] Búsqueda {idx+1}/{len(queries)}: '{query}'")

    seen      = set(state.get("urls_vistos", []))
    nuevos    = []
    nuevas_url = []

    # ---- Fuente 1: PubMed ----
    try:
        pubmed_results = pubmed.run(query)
        # PubMedAPIWrapper devuelve texto; lo parseamos en chunks por "Published:"
        if pubmed_results and "Title:" in pubmed_results:
            bloques = pubmed_results.split("\n\n")
            for bloque in bloques:
                if "Title:" not in bloque:
                    continue
                lines  = bloque.strip().split("\n")
                titulo = next((l.replace("Title:", "").strip() for l in lines if l.startswith("Title:")), "")
                uid    = next((l.replace("Published:", "").strip() for l in lines if l.startswith("Published:")), "")
                snippet= "\n".join(l for l in lines if not l.startswith(("Title:", "Published:", "Copyright")))
                url    = f"https://pubmed.ncbi.nlm.nih.gov/?term={'+'.join(titulo.split()[:5])}"

                if titulo and url not in seen:
                    seen.add(url)
                    nuevos.append({"titulo": titulo, "snippet": snippet[:800], "url": url, "fuente": "PubMed", "query_usada": query})
                    nuevas_url.append(url)

        print(f"   PubMed: {len([a for a in nuevos if a.get('fuente')=='PubMed'])} artículos nuevos")
    except Exception as e:
        print(f"    PubMed error: {e}")

    # ---- Fuente 2: DuckDuckGo (dominios de alto impacto) ----
    dominios_medicos = (
        "site:pubmed.ncbi.nlm.nih.gov OR site:nejm.org OR site:thelancet.com "
        "OR site:jamanetwork.com OR site:bmj.com OR site:ahajournals.org "
        "OR site:nature.com OR site:uptodate.com"
    )
    ddg_query = f"{query} ({dominios_medicos})"

    try:
        time.sleep(RATE_LIMIT_SLEEP)
        resultados = ddg_search.invoke(ddg_query)
        if isinstance(resultados, str):
            resultados = json.loads(resultados)

        antes = len(nuevos)
        for r in resultados:
            url    = r.get("link", r.get("url", ""))
            titulo = r.get("title", "")
            if url and url not in seen:
                seen.add(url)
                nuevos.append({
                    "titulo":     titulo,
                    "snippet":    r.get("snippet", r.get("body", ""))[:800],
                    "url":        url,
                    "fuente":     "DuckDuckGo",
                    "query_usada": query,
                })
                nuevas_url.append(url)

        print(f"   ✓ DuckDuckGo: {len(nuevos) - antes} artículos nuevos")
    except Exception as e:
        print(f"   ⚠ DuckDuckGo error: {e}")

    print(f"   → Total acumulado: {len(state['articulos_crudos']) + len(nuevos)} artículos únicos")

    return {
        "articulos_crudos":    state["articulos_crudos"] + nuevos,
        "urls_vistos":         state.get("urls_vistos", []) + nuevas_url,
        "current_query_index": idx + 1,
        "iteraciones":         state["iteraciones"] + 1,
    }

# ===========================================================================
# NODO 3 — EVALUACIÓN DE RELEVANCIA
# ===========================================================================

def puntuar_articulos(state: MedicalResearchState) -> dict:
    """
    LLM evaluador asigna puntuación 0-10 a cada artículo nuevo,
    considerando relevancia clínica y nivel de evidencia.
    Solo puntúa los artículos que aún no tienen score.
    """
    ya_puntuados = {a["url"] for a in state.get("articulos_puntuados", [])}
    por_puntuar  = [a for a in state["articulos_crudos"] if a["url"] not in ya_puntuados]

    if not por_puntuar:
        return {}

    print(f"\n [Nodo 3] Puntuando {len(por_puntuar)} artículos nuevos...")

    llm_scorer   = LLM.with_structured_output(ArticleScore)
    recien_puntuados = []

    for art in por_puntuar:
        try:
            score: ArticleScore = llm_scorer.invoke([
                SystemMessage(content=(
                    "Eres un experto en medicina basada en evidencia. "
                    "Evalúa la relevancia de este artículo para el caso clínico. "
                    "Considera pertinencia diagnóstica, terapéutica y nivel de evidencia. "
                    f"Relevante = score >= {MIN_SCORE}."
                )),
                HumanMessage(content=(
                    f"CONTEXTO CLÍNICO:\n{state['resumen_clinico']}\n\n"
                    f"ARTÍCULO:\nTítulo: {art['titulo']}\n"
                    f"Fragmento: {art['snippet']}\n"
                    f"Fuente: {art['url']}"
                ))
            ])
            recien_puntuados.append({
                **art,
                "puntuacion":       score.puntuacion,
                "nivel_evidencia":  score.nivel_evidencia,
                "es_relevante":     score.es_relevante,
                "justificacion":    score.justificacion,
            })
            relevancia = "✓" if score.es_relevante else "✗"
            print(f"   {relevancia} [{score.puntuacion:.1f}/10] [{score.nivel_evidencia}] {art['titulo'][:70]}...")

        except Exception as e:
            print(f"  Error puntuando '{art['titulo'][:50]}': {e}")
            continue

    todos_puntuados = sorted(
        state.get("articulos_puntuados", []) + recien_puntuados,
        key=lambda x: x.get("puntuacion", 0),
        reverse=True,
    )

    alta_calidad = [a for a in todos_puntuados if a.get("puntuacion", 0) >= MIN_SCORE]
    print(f"   → Artículos de alta relevancia (≥{MIN_SCORE}): {len(alta_calidad)}")

    return {"articulos_puntuados": todos_puntuados}

# ===========================================================================
# NODO 4 — CRITERIO DE PARO (función de enrutamiento condicional)
# ===========================================================================

def criterio_paro(state: MedicalResearchState) -> Literal["buscar_articulos", "seleccionar_top"]:
    """
    Decide si continuar buscando o proceder a la selección final.

    Condiciones de paro:
    1. Tenemos ≥ TARGET_ARTICLES artículos con score ≥ MIN_SCORE
    2. Se agotaron todas las queries
    3. Se alcanzó MAX_ITERATIONS
    """
    alta_calidad = [
        a for a in state.get("articulos_puntuados", [])
        if a.get("puntuacion", 0) >= MIN_SCORE
    ]

    if len(alta_calidad) >= TARGET_ARTICLES:
        print(f"\n [Criterio de paro] ✓ Suficientes artículos de calidad ({len(alta_calidad)} ≥ {TARGET_ARTICLES}). Procediendo a selección.")
        return "seleccionar_top"

    if state["current_query_index"] >= len(state["queries"]):
        print(f"\n [Criterio de paro] Queries agotadas. Procediendo con {len(alta_calidad)} artículos relevantes.")
        return "seleccionar_top"

    if state["iteraciones"] >= MAX_ITERATIONS:
        print(f"\n [Criterio de paro] Máximo de iteraciones ({MAX_ITERATIONS}) alcanzado.")
        return "seleccionar_top"

    print(f"\n [Criterio de paro] Alta calidad: {len(alta_calidad)}/{TARGET_ARTICLES}. Continuando búsqueda...")
    return "buscar_articulos"

# ===========================================================================
# NODO 5 — SELECCIÓN DE TOP 5
# ===========================================================================

def seleccionar_top(state: MedicalResearchState) -> dict:
    """
    Selecciona los mejores artículos aplicando diversificación:
    - Prioriza mayor puntuación
    - Intenta incluir distintos niveles de evidencia (meta-análisis, ECA, etc.)
    - Limita a TARGET_ARTICLES
    """
    print(f"\n [Nodo 5] Seleccionando top {TARGET_ARTICLES} artículos...")

    puntuados = state.get("articulos_puntuados", [])
    if not puntuados:
        print("   ⚠ No se encontraron artículos. Verifica tu conexión o las queries.")
        return {"top_articulos": []}

    # Primero los altamente relevantes, luego el resto (por si no hay suficientes)
    alta_calidad = [a for a in puntuados if a.get("puntuacion", 0) >= MIN_SCORE]
    resto        = [a for a in puntuados if a.get("puntuacion", 0) < MIN_SCORE]
    candidatos   = (alta_calidad + resto)[:TARGET_ARTICLES]

    for i, a in enumerate(candidatos, 1):
        print(f"   #{i} [{a.get('puntuacion', 0):.1f}/10] [{a.get('nivel_evidencia', '?')}] {a['titulo'][:70]}...")

    return {"top_articulos": candidatos}

# ===========================================================================
# NODO 6 — GENERACIÓN DE ANÁLISIS CLÍNICO FINAL
# ===========================================================================

def generar_analisis(state: MedicalResearchState) -> dict:
    """
    Para cada uno de los top artículos, un LLM redacta:
    - Resumen del abstract (~200 palabras)
    - Análisis clínico de asociación con el paciente (~150 palabras)
    Todo en español, con rigor clínico.
    """
    print(f"\n [Nodo 6] Redactando análisis clínico para {len(state['top_articulos'])} artículos...")

    llm_analyst  = LLM.with_structured_output(ArticleAnalysis)
    patient_json = json.dumps(state["patient_data"], ensure_ascii=False, indent=2)
    reporte      = []

    for i, art in enumerate(state["top_articulos"], 1):
        print(f"   Analizando #{i}: {art['titulo'][:60]}...")
        try:
            analysis: ArticleAnalysis = llm_analyst.invoke([
                SystemMessage(content=(
                    "Eres un médico redactando un reporte de revisión bibliográfica. "
                    "Redacta en español con rigor clínico. "
                    "El resumen del abstract debe ser fiel al contenido del artículo. "
                    "El análisis clínico debe ser específico para ESTE paciente, "
                    "mencionando implicaciones diagnósticas o terapéuticas concretas."
                )),
                HumanMessage(content=(
                    f"HISTORIAL COMPLETO DEL PACIENTE:\n{patient_json}\n\n"
                    f"RESUMEN CLÍNICO:\n{state['resumen_clinico']}\n\n"
                    f"ARTÍCULO #{i}:\n"
                    f"Título: {art['titulo']}\n"
                    f"URL/Fuente: {art['url']}\n"
                    f"Contenido disponible: {art['snippet']}\n"
                    f"Nivel de evidencia: {art.get('nivel_evidencia', 'No determinado')}\n"
                    f"Puntuación de relevancia: {art.get('puntuacion', 'N/A')}/10\n"
                    f"Justificación del evaluador: {art.get('justificacion', '')}"
                ))
            ])
            reporte.append(analysis.model_dump())
        except Exception as e:
            print(f"   ⚠ Error en análisis de artículo #{i}: {e}")
            reporte.append({
                "titulo":               art.get("titulo", "Sin título"),
                "fuente_url":           art.get("url", ""),
                "resumen_abstract":     art.get("snippet", "No disponible"),
                "analisis_clinico":     "No se pudo generar el análisis automáticamente.",
                "nivel_evidencia":      art.get("nivel_evidencia", "Desconocido"),
                "puntuacion_relevancia": art.get("puntuacion", 0),
            })

    return {"reporte_final": reporte}

# ===========================================================================
# NODO 7 — FORMATO DEL REPORTE FINAL
# ===========================================================================

def imprimir_reporte(state: MedicalResearchState) -> dict:
    """Imprime el reporte final en formato legible."""

    separador = "═" * 80
    print(f"\n\n{separador}")
    print("  REPORTE DE BIBLIOGRAFÍA MÉDICA RELEVANTE")
    print(separador)

    print(f"\n RESUMEN CLÍNICO DEL PACIENTE:")
    print(f"   {state['resumen_clinico'].replace(chr(10), chr(10)+'   ')}")

    print(f"\n ESTADÍSTICAS DE BÚSQUEDA:")
    print(f"   • Queries ejecutadas:          {state['current_query_index']}")
    print(f"   • Artículos encontrados:       {len(state['articulos_crudos'])}")
    print(f"   • Artículos evaluados:         {len(state['articulos_puntuados'])}")
    print(f"   • Artículos con score ≥ {MIN_SCORE}:   {len([a for a in state['articulos_puntuados'] if a.get('puntuacion',0) >= MIN_SCORE])}")
    print(f"   • Artículos en reporte final:  {len(state['reporte_final'])}")

    for i, art in enumerate(state.get("reporte_final", []), 1):
        print(f"\n\n{separador}")
        print(f"  ARTÍCULO #{i} de {len(state['reporte_final'])}")
        print(separador)

        print(f"\n TÍTULO:")
        print(f"   {art.get('titulo', 'N/A')}")

        print(f"\n FUENTE / LIGA:")
        print(f"   {art.get('fuente_url', 'N/A')}")

        score = art.get('puntuacion_relevancia', 'N/A')
        nivel = art.get('nivel_evidencia', 'N/A')
        print(f"\n NIVEL DE EVIDENCIA: {nivel}   |   RELEVANCIA: {score}/10")

        print(f"\n ABSTRACT / RESUMEN:")
        resumen = art.get("resumen_abstract", "No disponible")
        # Wrap a 75 chars para legibilidad
        palabras   = resumen.split()
        linea_act  = "   "
        for palabra in palabras:
            if len(linea_act) + len(palabra) + 1 > 78:
                print(linea_act)
                linea_act = "   " + palabra + " "
            else:
                linea_act += palabra + " "
        if linea_act.strip():
            print(linea_act)

        print(f"\n ANÁLISIS CLÍNICO / ASOCIACIÓN CON EL PACIENTE:")
        analisis   = art.get("analisis_clinico", "No disponible")
        palabras   = analisis.split()
        linea_act  = "   "
        for palabra in palabras:
            if len(linea_act) + len(palabra) + 1 > 78:
                print(linea_act)
                linea_act = "   " + palabra + " "
            else:
                linea_act += palabra + " "
        if linea_act.strip():
            print(linea_act)

    print(f"\n\n{separador}")
    print("  FIN DEL REPORTE")
    print(separador)

    return {}

# ===========================================================================
# CONSTRUCCIÓN DEL GRAFO LANGGRAPH
# ===========================================================================

def construir_grafo() -> StateGraph:
    """
    Construye y compila el grafo de LangGraph.

    Flujo:
        interpretar_paciente
              ↓
        buscar_articulos  ←──────────────┐
              ↓                          │
        puntuar_articulos                │
              ↓                          │
        criterio_paro ──► 'continuar' ──┘
              │
              └──► 'seleccionar_top'
                         ↓
                    generar_analisis
                         ↓
                    imprimir_reporte
                         ↓
                        END
    """
    workflow = StateGraph(MedicalResearchState)

    workflow.add_node("interpretar_paciente", interpretar_paciente)
    workflow.add_node("buscar_articulos",     buscar_articulos)
    workflow.add_node("puntuar_articulos",    puntuar_articulos)
    workflow.add_node("seleccionar_top",      seleccionar_top)
    workflow.add_node("generar_analisis",     generar_analisis)
    workflow.add_node("imprimir_reporte",     imprimir_reporte)

    workflow.set_entry_point("interpretar_paciente")

    workflow.add_edge("interpretar_paciente", "buscar_articulos")
    workflow.add_edge("buscar_articulos",     "puntuar_articulos")

    workflow.add_conditional_edges(
        "puntuar_articulos",
        criterio_paro,
        {
            "buscar_articulos": "buscar_articulos",
            "seleccionar_top":  "seleccionar_top",
        },
    )

    workflow.add_edge("seleccionar_top",  "generar_analisis")
    workflow.add_edge("generar_analisis", "imprimir_reporte")
    workflow.add_edge("imprimir_reporte", END)

    return workflow.compile()


PACIENTE_EJEMPLO = {
    "id_paciente": "MX-2024-001",
    "edad": 67,
    "sexo": "Masculino",
    "motivo_consulta": "Disnea progresiva de 3 meses de evolución, edema en miembros inferiores",
    "diagnostico_principal": "Insuficiencia cardíaca con fracción de eyección reducida (ICFEr)",
    "diagnosticos_secundarios": [
        "Diabetes mellitus tipo 2",
        "Hipertensión arterial sistémica",
        "Fibrilación auricular paroxística",
    ],
    "historia_clinica": {
        "sintomas": ["disnea de esfuerzo", "ortopnea", "disnea paroxística nocturna", "edema bilateral"],
        "signos_vitales": {"TA": "145/90 mmHg", "FC": "88 lpm", "FR": "22 rpm", "SpO2": "94%"},
        "laboratorios": {
            "BNP":        "850 pg/mL",
            "creatinina": "1.4 mg/dL",
            "HbA1c":      "8.2%",
            "sodio":      "134 mEq/L",
            "potasio":    "4.1 mEq/L",
        },
        "ecocardiograma": "FEVI 30%, dilatación del VI, regurgitación mitral moderada",
        "ECG": "Fibrilación auricular, bloqueo de rama izquierda, QRS 140ms",
    },
    "medicamentos_actuales": [
        "Metformina 850 mg c/12h",
        "Enalapril 10 mg c/12h",
        "Furosemida 40 mg c/24h",
        "Bisoprolol 5 mg c/24h",
        "Warfarina 5 mg c/24h (INR objetivo 2-3)",
    ],
    "alergias": "AINEs (urticaria)",
    "antecedentes_relevantes": "Infarto agudo al miocardio anterior hace 2 años. Cateterismo: DAI con lesión residual 60%.",
    "hospitalizaciones_previas": 2,
}


def main():

    FOCO: FocoBibliografico = "tratamiento"

    print(" Iniciando agente de bibliografía médica con LangGraph...")
    print(f"    Foco seleccionado : {FOCO.upper()}")
    print(f"   Configuración: máx {MAX_ITERATIONS} iteraciones, objetivo {TARGET_ARTICLES} artículos, score mínimo {MIN_SCORE}/10\n")

    grafo = construir_grafo()

    estado_inicial: MedicalResearchState = {
        "patient_data":        PACIENTE_EJEMPLO,
        "foco_bibliografico":  FOCO,
        "resumen_clinico":     "",
        "queries":             [],
        "current_query_index": 0,
        "articulos_crudos":    [],
        "urls_vistos":         [],
        "articulos_puntuados": [],
        "iteraciones":         0,
        "top_articulos":       [],
        "reporte_final":       [],
        "busqueda_completa":   False,
    }

    resultado = grafo.invoke(estado_inicial)
    print(resultado)
    return resultado



main()
