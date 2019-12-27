FROM bedore/nlp-modules-base
COPY . /app
WORKDIR /app
RUN pip install poetry && poetry install -E all
