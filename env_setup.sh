#!/usr/bin/env bash

pip install --upgrade pip
cat requirements.txt | xargs -n 1 pip install
pip install gpustat
pip install tensorflow-gpu==2.0
pip install -e .

git config --global user.name "Faizan Ahemad"
git config --global user.email fahemad3@gmail.com

# install tf
# install fasttext