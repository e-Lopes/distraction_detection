FROM python:3.11-slim

WORKDIR /app

COPY docker/analysis/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/

ENV PYTHONPATH=/app

# Por padrão, roda agregação + gráficos em sequência.
ENTRYPOINT ["sh", "-c", "python -m src.analysis.aggregate --results-dir /results && python -m src.analysis.plots --results-dir /results"]
