#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi
mkdir -p venvs
venvdir=venvs/$(date +"%y-%m-%d-%H-%M")
python3.7 -m venv $venvdir
source $venvdir/bin/activate
pip install $1
python test_package.py $1
