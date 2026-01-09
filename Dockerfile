# ===== Base image =====
FROM python:3.12-slim

# ===== Set working directory =====
WORKDIR /app

# ===== Install system dependencies =====
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git && \
    rm -rf /var/lib/apt/lists/*

# ===== Upgrade pip & install uv =====
RUN pip install --upgrade pip
RUN pip install uv

# ===== Copy only dependency files first =====
COPY pyproject.toml uv.lock* ./

# ===== Install Python dependencies with uv =====
RUN uv pip install --system --locked

# ===== Copy the rest of your app =====
COPY . .

# ===== Expose Streamlit port =====
EXPOSE 8501

# ===== Run the app =====
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
