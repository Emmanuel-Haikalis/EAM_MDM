# Stage 1: build dependencies
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# Stage 2: runtime image
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="Asset Classification Tool"
LABEL org.opencontainers.image.description="Classifies asset registers against standardised classification systems."

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source
COPY src/ ./src/
COPY config/ ./config/
COPY main.py ./

# Default output directory
RUN mkdir -p /data/output

VOLUME ["/data"]

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
