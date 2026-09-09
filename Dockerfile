FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for building certain Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry

# Configure poetry to not create a virtual environment inside Docker
RUN poetry config virtualenvs.create false

# Copy poetry dependency definitions
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry install --no-root --no-interaction --no-ansi

# Copy project files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Default environment variables
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Healthcheck for Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

