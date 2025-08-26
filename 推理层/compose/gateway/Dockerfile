FROM python:3.10-slim

WORKDIR /app
COPY ../../12_实战_OpenAI兼容API网关_FastAPI.md /tmp/doc.md

# 复制最小网关代码
COPY gateway_app /app

RUN pip install --no-cache-dir fastapi uvicorn httpx==0.27.0 pydantic==2.* aiolimiter

EXPOSE 9000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]


