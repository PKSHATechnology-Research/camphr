.PHONY: lint
LIB=camphr

lint:
	isort ${LIB}
	black ${LIB}

