# genRBF-missing

This package implements **genRBF** (see [Generalized RBF kernel for incomplete data](https://arxiv.org/abs/1612.01480)). The code is written in Python and Cython.


## Requirements:
* numpy (1.12.1 or higher),
* cython (0.25.2 or higher),
* gcc, g++ (5.4.0 or higher). 

To estimate a Gaussian density from incomplete data used in genRBF, we applied **R** package [norm](https://cran.r-project.org/web/packages/norm/index.html).

## Installation

Go to directory [genRBF-missing/genRBF_source/](https://github.com/struski2/genRBF-missing/tree/master/genRBF_source) and run the following instruction in terminal:

```
./build.sh
```
or 
```
python setup.py build_ext --inplace
```

## Usage

The file '[main_demo.py](https://github.com/struski2/genRBF-missing/blob/master/main_demo.py)' applies genRBF to SVM classifier. To run this demo, type the following command in terminal:
```
./main_demo.py ./data/
```
A directory [genRBF-missing/data/](https://github.com/struski2/genRBF-missing/tree/master/data/) contains exemplary data with missing attributes: *train data*, *test data* and their labels. Files *mu.txt*, *cov.txt* contain covariance matrix and mean estimated over train data and were created with use of [norm.R](https://github.com/struski2/genRBF-missing/blob/master/norm.R):
```
./norm.R ./data/train_data.txt ./data/
```
