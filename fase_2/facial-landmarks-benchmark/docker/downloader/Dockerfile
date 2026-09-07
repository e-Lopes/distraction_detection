FROM python:3.11-slim
RUN pip install --no-cache-dir kaggle==1.6.17
COPY src/ /app/src/
WORKDIR /app
ENTRYPOINT ["python", "-m", "src.datasets.download_public"]
