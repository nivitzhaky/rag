FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

CMD ["sh", "-c", "python -c 'from vector_store import create_vector_store; create_vector_store()' && uvicorn api:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app"]
