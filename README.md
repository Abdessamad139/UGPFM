# UGPFM

This is the official implementation of the UNSUPERVISED GRAPH PREDICTOR FACTORIZATION MACHINE (UGPFM) algorithm. This new approach was publised in Journal of King Saud University - Computer and Information Sciences (JKSUCIS).

By Abdessamad Chanaa & Nour-eddine El Faddouli.

#### Related Paper

An Analysis of learners’ affective and cognitive traits in Context-Aware Recommender Systems (CARS) using feature interactions and Factorization Machines (FMs).

## Datasets

This folder contains the two used dataset (after cleaning and process based on the chosen contexts) : 

	1- coursera: contains "100K Coursera's Course Reviews dataset"
	2- level: contains "Coursera Course Dataset"
  
## Files

**./coursera/review_complete_data_final.csv** : contains "100K Coursera’s Course Reviews" dataset.

**./coursera/data_difficulty_user.csv** : contains "Coursera Course Dataset" coursera dataset.

**./UGPFM_FAST/pylibfm**: the official implementation of FM based on (https://github.com/coreylynch/pyFM).

**./UGP/UGP.py**: the implementation of UGP that generates feature interactions. it is based on (https://github.com/zhuangAnjun/Glomo).

**setup.py** : the build file for cython it contains the compilations options to produce the .c file from the .pyx file.

**UGPFM_fast.pyx**: Cython file. it will compiled automatically to a .c file thanks to "setup.py"

**UGPFM.py**: the main file that combine UGP with FM.


## Requirements

1- Python 3.7 

2- TensorFlow 2.4.0

3- Cython 0.29.12

## USE

To launch the implementation, go to the downloaded folder then execute:

	python3 setup.py build_ext --inplace
	python3 UGPFM.py --lr 0.001 --iter 100 --fact 10
 
## Citation

If you use those data or the framework please cite us:

