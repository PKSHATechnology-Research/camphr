FROM docker.pkg.github.com/tamuhey/dockerfiles/mecab:ci-build
ENV POETRY_VERSION=1.1.5
RUN pip install -U poetry==${POETRY_VERSION}
RUN poetry config virtualenvs.create false

WORKDIR /app
COPY . .
