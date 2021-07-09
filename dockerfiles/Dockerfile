ARG PYTHON_VERSION
FROM python:${PYTHON_VERSION}-buster

# install python dependencies
WORKDIR /app
COPY pyproject.toml /app
RUN pip install -U poetry && \
    poetry config virtualenvs.in-project false && \
    poetry config virtualenvs.path ~/.virtualenvs
RUN poetry update && poetry install --no-root -E mecab

# install source
COPY . /app
RUN poetry install
