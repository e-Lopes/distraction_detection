FROM python:3.10-slim

# Variável de ambiente para permitir GUI
ENV DISPLAY=:0

# Atualiza o sistema e instala dependências do sistema
RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Cria diretório de trabalho
WORKDIR /app

# Copia os arquivos para dentro do container
COPY . .

# Instala dependências Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão
CMD ["python", "main.py"]
