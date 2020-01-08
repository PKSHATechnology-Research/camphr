# Contributing

- 開発にあたり，spacyについて以下のドキュメントを読むことをお勧めします
    - [Architecture · spaCy API Documentation](https://spacy.io/api)
    - [Saving and Loading · spaCy Usage Documentation](https://spacy.io/usage/saving-loading)

## setup

1. clone
2. `$ poetry install -E all`
3. `$ pre-commit install`
4. `$ poetry run pytest tests`

## 構成

- `camphr`: package source
- `tests`: test files
- `scripts`: 細々としたスクリプト. packagingに必要なものなど

