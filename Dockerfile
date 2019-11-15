FROM bedore/nlp-modules-base
COPY . /app
WORKDIR /app
RUN pip install pipenv && PIPENV_SKIP_LOCK=1 pipenv install --dev && pipenv run pip install -e .
