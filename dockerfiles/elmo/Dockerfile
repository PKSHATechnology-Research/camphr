FROM python:3.7.6

COPY . /app
WORKDIR /app
RUN pip install -U poetry
RUN poetry run python scripts/download_model.py en_elmo_medium
