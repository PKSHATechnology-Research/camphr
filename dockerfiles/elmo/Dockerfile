FROM python:3.7.6

ARG GITHUB_TOKEN

COPY . /app
WORKDIR /app
RUN pip install -U poetry github_asset && poetry install -E allennlp && make download_elmo
