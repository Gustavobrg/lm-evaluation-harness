FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema e Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Define diretório de trabalho
WORKDIR /app

# Atualiza pip e ferramentas de build
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Copia o código local para dentro do container (pode ignorar se você for montar via -v)
COPY . /app

# Instala lm-eval e libs adicionais
RUN pip3 install -e . && \
    pip3 install transformers sae_lens sae-dashboard

CMD ["bash"]
