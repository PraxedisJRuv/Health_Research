# Medical Research Tool
An Agent that given a clinical record and a focus (diagnosis, treatment, epidemiology), can find the best bibliographical references suited fot the patient. Intended to be a part of an application for physcicians who like to be updated in their consult.

## Features
- Process clinical records in json format to output bibliographical references and a summary on it's relation to the patient.
- PubMed API avialable for consult (if user has a PubMed account).
- Search made by DuckDuckGo engine in the websites with more prestige for medical publications.
- Highly and easily customizable (llm slection and search parameters)

## Tech Stack
- **Programming Language**: Python
- **Libraries**: OpenAI, Requests, FastAPI
- **APIs**: PubMed API, DuckDuckGo Search API
- **Environment**: Virtual Environment (.venv)

## Installation
1. Clone the repository:
    bash
   git clone <repository-url>
   
2. Navigate to the project directory:
    bash
   cd Health_hackathon
   
3. Create and activate a virtual environment:
    bash
   python -m venv .venv
   .venv\Scripts\activate
   
4. Install the required dependencies:
    bash
   pip install -r requirements.txt
  

## Usage
1. Ensure the virtual environment is activated:
    bash
   .venv\Scripts\activate
   
2. Run the main script:
    bash
   python learn.py
   
3. Follow the prompts to input clinical records and specify the focus (diagnosis, treatment, epidemiology).

## Project Structure
```
Health_hackathon/
├── Agent/
│   ├── edges.py       # Handles graph edges
│   ├── graph.py       # Core graph logic
│   ├── nodes.py       # Handles graph nodes
│   ├── schema.py      # Defines data schemas
│   └── test.py        # Unit tests for the Agent module
│           
├── README.md          
└── requirements.txt   
```
