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


# Experiments
- Content Based
- Collaborative NMF Matrix Based from surprise
- Content + Collaborative NMF
- Content + Collaborative with extra features
- Content + Collaborative with extra features with alpha tree

# Innovation
- Heterogenous Features via Deep Networks
- Weighted Triplet Loss
- Embedding Compression
    - We train in a higher Dimensional Space, After training we use autoencoders to reduce dimensionality. 
    - Since our task involves cosine distance, after auto-enc step we do another step where we use triplet loss with Distances calculated from initial bigger embeddings. 
    This is similar to TSNE.
    - the two steps can be combined into one encoder-decoder-triplet architecture where decoder loss and triplet loss are weighted and added.
    
- Combine Collaborative and Content Based Approach by 
    - building content embeddings first
    - enhancing them with collaborative relations
    - Balancing between them using a weighted scheme to solve cold start problem
- Multiple hybrid embeddings for sellers at different life-cycle stages. Multiple alpha


# References

### Misc References
    - http://stevehanov.ca/blog/?id=145
    
### Dimensionality Reduction
    - https://github.com/DmitryUlyanov/Multicore-TSNE
    - https://github.com/lmcinnes/umap
    - https://github.com/KlugerLab/FIt-SNE https://pypi.org/project/fitsne/0.1.10/
    - https://stats.stackexchange.com/questions/402668/intuitive-explanation-of-how-umap-works-compared-to-t-sne
    

 

