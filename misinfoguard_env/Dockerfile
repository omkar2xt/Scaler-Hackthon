FROM python:3.11-slim@sha256:9358444059ed78e2975ada2c189f1c1a3144a5dab6f35bff8c981afb38946634

RUN groupadd --gid 1000 appgroup && useradd --uid 1000 --gid appgroup --create-home appuser

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chown -R appuser:appgroup /app

EXPOSE 7860

USER appuser

CMD ["python", "inference.py"]
