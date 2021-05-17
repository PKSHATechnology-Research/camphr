.PHONY: lint test
LIB=camphr

lint:
	isort ${LIB}
	black ${LIB}

test:
	poetry run mypy ${LIB}
	poetry run flake8 ${LIB}
	poetry run pytest tests
