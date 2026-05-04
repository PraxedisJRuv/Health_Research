# Medical Research Tool
An Agent that given a clinical record and a focus (diagnosis, treatment, epidemiology), can find the best bibliographical references suited for the patient. Intended to be a part of an application for physicians who like to be updated in their consult.

## Features
- Process clinical records in JSON format to output bibliographical references and a summary on its relation to the patient.
- PubMed API available for consult (if user has a PubMed account).
- Search made by DuckDuckGo engine in the websites with more prestige for medical publications.
- Highly and easily customizable (LLM selection and search parameters)

## Tech Stack
- **Programming Language**: Python
- **Libraries**: FastAPI, Uvicorn, LangChain, SQLModel, Pydantic
- **Model provider**: Google Gemini via `langchain-google-genai`
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

2. Start the backend API server:
   ```bash
   uvicorn Backend.app:app --reload
   ```

3. Open the API docs at:
   ```text
   http://127.0.0.1:8000/docs
   ```

4. Use the API endpoints to submit patient records, create consults, and generate analysis results.

## Docker Usage
1. Build the Docker image:
   ```bash
   docker build -t health-backend .
   ```

2. Run the backend container:
   ```bash
   docker run -p 8000:8000 health-backend
   ```

3. Visit the API docs at:
   ```text
   http://127.0.0.1:8000/docs
   ```

> If you already have Docker installed, this lets you run the backend without creating a local Python virtual environment.

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
