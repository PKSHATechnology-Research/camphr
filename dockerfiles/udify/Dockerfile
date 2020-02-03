FROM python:3.7.6

ARG GITHUB_TOKEN

COPY . /app
WORKDIR /app
RUN pip install -U poetry github_asset
RUN make download_udify
RUN poetry install -E udify
