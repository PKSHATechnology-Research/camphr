#!/bin/bash

for d in $(ls -d */); do
    cd $d && poetry update && cd ..
done
