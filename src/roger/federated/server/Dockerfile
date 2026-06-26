# Lean image for the aggregation server. It needs only torch (CPU — the server does NO model
# inference, just int64 sums + safetensors I/O), safetensors, cryptography, fastapi, uvicorn — NOT
# the full `roger` dependency set (transformers/peft/bitsandbytes are client/training-only). So we
# copy the source and set PYTHONPATH instead of `pip install .`.
#
# Build from the REPO ROOT:   docker build -f src/roger/federated/server/Dockerfile -t roger-agg .
# Run:                         docker run -p 8000:8000 -v roger-agg-data:/data -e ROGER_SERVER_DATA=/data roger-agg
FROM python:3.13-slim

RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir safetensors cryptography fastapi "uvicorn[standard]"

WORKDIR /app
COPY src/ /app/src/
ENV PYTHONPATH=/app/src
ENV ROGER_SERVER_DATA=/data
EXPOSE 8000

CMD ["python", "-m", "roger.federated.server"]
