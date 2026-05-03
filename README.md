# Medical Research Tool
An Agent that given a clinical record and a focus (diagnosis, treatment, epidemiology), can find the best bibliographical references suited for the patient. Intended to be a part of an application for physicians who like to be updated in their consult.

## Features
- Process clinical records in JSON format to output bibliographical references and a summary on its relation to the patient.
- PubMed API available for consult (if user has a PubMed account).
- Search made by DuckDuckGo engine in the websites with more prestige for medical publications.
- Highly and easily customizable (LLM selection and search parameters)

## Tech Stack
- **Programming Language**: Python
- **Libraries**: OpenAI, Requests, FastAPI
- **APIs**: PubMed API, DuckDuckGo Search API
- **Environment**: Virtual Environment (.venv)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd Health_hackathon
   ```

3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Ensure the virtual environment is activated:
   ```bash
   .venv\Scripts\activate
   ```

2. Run the main application:
   ```bash
   python Backend/app.py
   ```
   for propper API use and documentation:
   ```bash
   uvicorn Backend\app:app --reload
   ```

3. Follow the prompts to input clinical records and specify the focus (diagnosis, treatment, epidemiology).

## Project Structure
```
Health_hackathon/
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── Backend/
│   ├── __init__.py
│   ├── app.py           # Main FastAPI application
│   ├── models.py        # Data models
│   ├── utils.py         # Utility functions
│   └── Agent/
│       ├── __init__.py
│       ├── edges.py     # Handles graph edges
│       ├── graph.py     # Core graph logic
│       ├── nodes.py     # Handles graph nodes
│       ├── schema.py    # Defines data schemas
│       └── test.py      # Unit tests for the Agent module
```
