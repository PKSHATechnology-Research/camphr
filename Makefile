.PHONY: lint
LIB=camphr

lint:
	autoflake --remove-all-unused-imports --in-place -r ${LIB}
	isort ${LIB}
	black ${LIB}

