# Learn Survival Model with Pysurvival

<center><img src="docs/pages/images/py_survival_logo" alt="dsciencelabs_logo" title="dsciencelab_logo" width="50%", height="50%" /></center>

## What is Pysurvival ?

PySurvival is an open source python package for Survival Analysis modeling - *the modeling concept used to analyze or predict when an event is likely to happen*. It is built upon the most commonly used machine learning packages such [NumPy](http://www.numpy.org/), [SciPy](https://www.scipy.org/) and [PyTorch](https://pytorch.org/).

PySurvival is compatible with Python 2.7-3.7.

Check out the documentation [here](https://dsciencelabs.github.io/survivalmodel_py/)

---

## Installation

If you have already installed a working version of gcc, the easiest way to install Pysurvival is using pip
```
pip install pysurvival
```
The full description of the installation steps can be found [here](https://www.pysurvival.io/installation.html).

---

## Get Started

Because of its simple API, Pysurvival has been built to provide to best user experience when it comes to modeling.
Here's a quick modeling example to get you started:

```python
# Loading the modules
from pysurvival.models.semi_parametric import CoxPHModel
from pysurvival.models.multi_task import LinearMultiTaskModel
from pysurvival.datasets import Dataset
from pysurvival.utils.metrics import concordance_index

# Loading and splitting a simple example into train/test sets
X_train, T_train, E_train, X_test, T_test, E_test = \
	Dataset('simple_example').load_train_test()

# Building a CoxPH model
coxph_model = CoxPHModel()
coxph_model.fit(X=X_train, T=T_train, E=E_train, init_method='he_uniform', 
                l2_reg = 1e-4, lr = .4, tol = 1e-4)

# Building a MTLR model
mtlr = LinearMultiTaskModel()
mtlr.fit(X=X_train, T=T_train, E=E_train, init_method = 'glorot_uniform', 
           optimizer ='adam', lr = 8e-4)

# Checking the model performance
c_index1 = concordance_index(model=coxph_model, X=X_test, T=T_test, E=E_test )
print("CoxPH model c-index = {:.2f}".format(c_index1))

c_index2 = concordance_index(model=mtlr, X=X_test, T=T_test, E=E_test )
print("MTLR model c-index = {:.2f}".format(c_index2))
```

---


