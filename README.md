# UGPFM

This is the official implementation of the UNSUPERVISED GRAPH PREDICTOR FACTORIZATION MACHINE (UGPFM) algorithm. This new approach was published in the Journal of King Saud University - Computer and Information Sciences (JKSUCIS).

By Abdessamad Chanaa & Nour-eddine El Faddouli.

#### Related Paper

An Analysis of learners’ affective and cognitive traits in Context-Aware Recommender Systems (CARS) using feature interactions and Factorization Machines (FMs).

## Datasets

This folder contains the two used datasets (after cleaning and process based on the chosen contexts) : 

	1- Coursera: contains "100K Coursera's Course Reviews dataset"
	2- Level: contains "Coursera Course Dataset"
  
## Files

**```./coursera/review_complete_data_final.csv```** : contains "100K Coursera’s Course Reviews" dataset.

**```./level/data_difficulty_user.csv```** : contains "Coursera Course Dataset" dataset.

**```./UGPFM_FAST/pylibfm```**: the official implementation of FM based on (https://github.com/coreylynch/pyFM).

**```./UGP/UGP.py```**: the implementation of UGP that generates feature interactions. it is based on (https://github.com/zhuangAnjun/Glomo).

**```setup.py```** : the build file for Cython. It contains the compilations options to produce the .c file from the .pyx file.

**```UGPFM_fast.pyx```**: Cython file, it will compiled automatically to a .c file thanks to "setup.py"

**```UGPFM.py```**: the main file that combines UGP with FM.


## Requirements

1- Python 3.7 

2- TensorFlow 2.4.0

3- Cython 0.29.12

## Usage:

	git clone https://github.com/Abdessamad139/UGPFM.git
	cd UGPFM/
	python3 setup.py build_ext --inplace
	python3 UGPFM.py --lr 0.001 --iter 100 --fact 10
 
## Citation

If you use these data or the framework, please cite us:

```
Abdessamad Chanaa, Nour-eddine El Faddouli, An Analysis of learners’ affective and cognitive traits in Context-Aware Recommender Systems (CARS) using feature interactions and Factorization Machines (FMs), Journal of King Saud University - Computer and Information Sciences, 2021, ISSN 1319-1578, https://doi.org/10.1016/j.jksuci.2021.06.008.
```
or BibTex

```

@article{chanaa2022analysis,
  title={An Analysis of learners’ affective and cognitive traits in Context-Aware Recommender Systems (CARS) using feature interactions and Factorization Machines (FMs)},
  author={Chanaa, Abdessamad and others},
  journal={Journal of King Saud University-Computer and Information Sciences},
  volume={34},
  number={8},
  pages={4796--4809},
  year={2022},
  publisher={Elsevier}
}

```

