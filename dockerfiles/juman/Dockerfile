FROM docker.pkg.github.com/tamuhey/dockerfiles/knp:latest
ENV POETRY_VERSION=1.1.5
RUN pip install -U poetry==${POETRY_VERSION}

WORKDIR /app
COPY . .
