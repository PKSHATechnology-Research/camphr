FROM docker.pkg.github.com/tamuhey/dockerfiles/knp:latest
COPY . /app
WORKDIR /app
RUN pip install -U poetry && poetry update && poetry install -E juman
