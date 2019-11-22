#!/usr/bin/env bash

pip install --upgrade pip
pip install -r requirements.txt
cat requirements.txt | xargs -n 1 pip install
pip install gpustat

# install tf
# install fasttext