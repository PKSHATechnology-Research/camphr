#!/bin/bash

set -e
set -v

poetry run mypy $1
poetry run flake8 $1
poetry run pytest tests
