FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY uv.lock* .

# Install dependencies with uv
RUN uv pip install --system numpy pandas scikit-learn xgboost streamlit plotly scipy pyarrow statsmodels

# Copy application code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]