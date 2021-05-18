#!/bin/bash

set -e

if [[ -z $1 ]]; then
    >&2 echo "package name not specified"
    exit 1
fi

function test_package() {
    poetry run mypy $1
    poetry run flake8 $1
    poetry run pytest tests
}

set -x

if [[ $1 == "all" ]]; then
    cd ./subpackages/camphr_cli && test_package camphr_cli && cd ../..
    test_package camphr
else
    test_package $1
fi
