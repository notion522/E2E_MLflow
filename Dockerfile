# ✅ Updated Dockerfile
FROM python:3.10-slim-bullseye

# Install system dependencies and awscli
RUN apt-get update -y && apt-get install -y awscli

WORKDIR /app
COPY . /app

# Always upgrade pip before installing dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
