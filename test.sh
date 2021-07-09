#!/bin/bash

set -e
set -v

PYTHON_VERSION=$1
if [[ -z $PYTHON_VERSION ]]; then
  echo "Python version should be specified"
  exit 1
fi

EXTRA=$2
if [[ -z $EXTRA ]]; then
  echo "Extra should be specified"
  exit 1
fi

docker build -f dockerfiles/Dockerfile.${EXTRA} \
    --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
    -t camphr_${PYTHON_VERSION}_${EXTRA} \
    .
