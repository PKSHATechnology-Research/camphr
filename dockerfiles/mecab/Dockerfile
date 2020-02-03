FROM tamuhey/mecab:py3.7.6-ipadic-1.0.1
COPY . /app
WORKDIR /app
RUN pip install -U poetry && poetry install -E mecab
