FROM python:3.10-slim

# Evita que apt-get pida confirmaciones manuales
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
# Se cambió 'libgl1-mesa-glx' por 'libgl1' que es el estándar actual
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxrender1 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario vscode
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

WORKDIR /workspace

# Instalar requerimientos si existen
COPY requirements.txt* /tmp/pip-tmp/
RUN if [ -f /tmp/pip-tmp/requirements.txt ]; then \
      pip install --no-cache-dir -r /tmp/pip-tmp/requirements.txt; \
    fi \
    && rm -rf /tmp/pip-tmp

USER $USERNAME