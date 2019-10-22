# Hybrid-Weighted-Embedding-Recommender
A Hybrid Recommendation system which uses Content embeddings and augments them with collaborative features. Weighted Combination of embeddings enables solving cold start with fast training and serving

# Environment Setup
- Install Anaconda from https://www.anaconda.com/distribution/

Add `.condarc` to your home dir with below contents

```bash
auto_update_conda: False
channels:
  - defaults
  - anaconda
  - conda-forge
always_yes: True
add_pip_as_python_dependency: True
use_pip: True
create_default_packages:
  - pip
  - ipython
  - jupyter
  - nb_conda
  - setuptools
  - wheel
```

`conda update conda`

`conda create -n hybrid-recsys python=3.7.4`

`conda activate hybrid-recsys`


Install [Fasttext](https://fasttext.cc/docs/en/supervised-tutorial.html)

```bash
wget https://github.com/facebookresearch/fastText/archive/v0.9.1.zip
unzip v0.9.1.zip
cd fastText-0.9.1 && make -j4 && pip install .
```

Install Tensorflow 2.0 from [here](https://www.tensorflow.org/install)
```bash
pip install --upgrade pip
pip install tensorflow
```

pip install -r requirements.txt

