FROM bedore/nlp-modules-base
COPY . /app
WORKDIR /app
RUN pipenv install --dev && pipenv install sklearn
