FROM python:3.7.6

ARG GITHUB_TOKEN

COPY . /app
WORKDIR /app
RUN pip install -U poetry github_asset && make download_udify && poetry install -E udify 
