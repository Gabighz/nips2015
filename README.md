NIPS 2015
=========

Code for reproducing the key results of our NIPS 2015 paper on semi-supervised low-rank logistic regression models for large functional neuroimaging datasets.

Bzdok D, Eickenberg M, Grisel O, Thirion B, Varoquaux G.
**Semi-supervised Factored Logistic Regression for High-Dimensional Neuroimaging Data**
Advances in Neural Information Processing Systems (**NIPS 2015**), Montreal.
[Paper on ResearchGate](https://www.researchgate.net/publication/281490102_Semi-Supervised_Factored_Logistic_Regression_for_High-Dimensional_Neuroimaging_Data)

Please cite this paper when using the code for your research.

To follow established conventions of scikit-learn estimators, the ``SSEncoder`` class exposes the functions fit(), predict(), and score().
This should allow for seamless integration into other scikit-learn-enabled machine-learning pipelines.

For questions and bug reports, please send me an e-mail at _danilobzdok[at]gmail.com_.

## Prerequisites

1. Development setup from a directory above this repository's clone, assuming a clean installation in Ubuntu:

    ```
    sudo apt install python3-dev
    sudo apt install python3-tk
    sudo apt install virtualenv
    virtualenv nips2015 -p python3
    cd nips2015/
    source bin/activate
    pip install scikit-learn numpy scipy nibabel nilearn theano matplotlib joblib pandas
    python download_data.py
    python nips3mm.py
    ```

    Now the files are ready to be executed!



