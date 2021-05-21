FROM docker.pkg.github.com/tamuhey/dockerfiles/mecab:ci-build
WORKDIR /app
ENV POETRY_VERSION=1.1.5
RUN pip install -U poetry==${POETRY_VERSION} && poetry update && poetry install -E mecab
COPY . .
