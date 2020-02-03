FROM bedore/nlp-modules-base:1.0.0
COPY . /app
WORKDIR /app
RUN pip install -U pip
RUN pip install -U setuptools
RUN pip install -U poetry
RUN poetry install -E all
