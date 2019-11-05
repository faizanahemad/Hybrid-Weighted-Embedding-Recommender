from setuptools import find_packages, setup

setup(name='hwer',
      version='0.0.1',
      description='',
      url='https://github.com/faizanahemad/Hybrid-Weighted-Embedding-Recommender',
      author='Faizan Ahemad',
      author_email='fahemad3@gmail.com',
      license='MIT',
      install_requires=[
          'numpy', 'pandas', 'beautifulsoup4', 'more-itertools',
          'dill', 'pytidylib', 'seaborn', 'gensim', 'nltk','scikit-learn',
          'joblib', 'tensorflow', 'tqdm', 'nmslib', 'flair',
      ],
      keywords=['data-science', 'ML', 'Machine Learning', 'Recommendation System', 'Hybrid Recommendation System', 'Embeddings'],
      packages=find_packages(),
      test_suite=None,
      tests_require=None,
      zip_safe=False)
