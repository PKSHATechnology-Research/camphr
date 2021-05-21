FROM python:3.8-slim
RUN apt update && apt install build-essential -y
RUN pip install -U poetry==1.1.5

WORKDIR /app
COPY . .
