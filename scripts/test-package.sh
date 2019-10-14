#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi
python3.7 -m venv .venv
source .venv/bin/activate
pip install $1
python test_package.py $1
