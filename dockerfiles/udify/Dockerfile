FROM python:3.7.6

COPY . /app
WORKDIR /app
RUN python scripts/download_model.py en_udify
RUN pip install pytest hypothesis
