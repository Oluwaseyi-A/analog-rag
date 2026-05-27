FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps + ngspice 42 (Ubuntu 24.04 default) + Node.js 22 LTS
# Ubuntu 24.04 ships python3.12 as 'python3' — no python3.11 package available
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    ngspice \
    libngspice0 \
    libngspice0-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
       | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
       > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Ensure 'python' resolves to python3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Claude Code CLI
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY src/ ./src/

EXPOSE 8501

CMD ["streamlit", "run", "src/ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
