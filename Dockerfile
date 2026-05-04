FROM python:3.12-slim

WORKDIR /app

RUN python -m pip install --upgrade pip

COPY requirements.txt ./
RUN pip install --no-cache-dir fastapi uvicorn sqlmodel pydantic python-dotenv langchain-google-genai langchain-core langchain-community langgraph ddgs xmltodict

COPY . .

EXPOSE 8000

CMD ["uvicorn", "Backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
