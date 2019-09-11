FROM bedore/nlp-modules-base
COPY . /app
WORKDIR /app
RUN pip install pipenv awscli && pipenv install --system --dev
