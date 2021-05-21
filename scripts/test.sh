#!/bin/bash

set -e

if [[ -z $1 ]]; then
    >&2 echo "ERROR: package name not specified"
    exit 1
fi
if [[ "$2" ]]; then
  extras=$2
fi

function test_package() {
    poetry run mypy $1
    poetry run flake8 $1
    poetry run pytest tests
}

set -x

if [[ $1 == "all" ]]; then
    for subpackage in camphr_embedrank camphr_cli camphr_pattern_search; do
        cd ./subpackages/${subpackage} 
        if [[ "$extras" ]]; then
          poetry update $extras
        fi
        test_package ${subpackage} 
        cd ../..
    done;
    test_package camphr
else
    test_package $1
fi
