##############################
# Stage 1 — Build dependencies
##############################
FROM python:3.10-slim AS builder
# Python 3.10 chosen for maximum ML ecosystem compatibility

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Build wheels for all dependencies
RUN python -m pip install --upgrade pip wheel setuptools && \
    pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


##############################
# Stage 2 — Final minimal image
##############################
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies from wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Copy application code
COPY --chown=appuser:appuser streamlit_app.py /app/streamlit_app.py
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser final_model/ /app/final_model/

# Ensure model directory exists
RUN mkdir -p /app/final_model

EXPOSE 8501

# Streamlit environment (cleaner than long CMD flags)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

CMD ["streamlit", "run", "streamlit_app.py"]
