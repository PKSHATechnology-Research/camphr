#!/bin/bash
set -e

poetry install
poetry run mypy camphr_lang
poetry run flake8 camphr_lang
poetry run pytest tests

