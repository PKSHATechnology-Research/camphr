FROM python:3.7.6

COPY . /app
WORKDIR /app
RUN python scripts/download_model.py en_elmo_medium
RUN pip install pytest hypothesis toml
