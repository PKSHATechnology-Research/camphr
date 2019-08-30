FROM bedore/nlp-modules-base
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
