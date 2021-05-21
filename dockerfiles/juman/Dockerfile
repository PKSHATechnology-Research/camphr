FROM docker.pkg.github.com/tamuhey/dockerfiles/knp:latest
WORKDIR /app
ENV POETRY_VERSION=1.1.5
RUN pip install -U poetry==${POETRY_VERSION} && poetry update && poetry install -E juman
