FROM bedore/nlp-modules-base
COPY . /app
WORKDIR /app
RUN pip install pipenv && pipenv install --dev && pipenv install -e .
