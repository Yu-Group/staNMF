Yu Wang, 2018

staNMF
------
Python 3 implementation of `Siqi Wu et al.'s 2016 stability NMF (staNMF)
<http://doi.org/10.1073/pnas.1521171113>`_

Package Contents
----------------

=========
main.py
=========
Contains the major classes.

=================
Demo/01_Drosophila.ipynb
=================
Example of staNMF demonstrated on Wu et al.'s 2016
drosophila spatial expression data between K=15 and K=30; Generates
sample factorizations, calculates instability index, and plots instability
against K

============================
Demo/data/WuExampleExpression.csv
============================
sample dataset (also available for download `here
<http://insitu.fruitfly.org/cgi-bin/ex/insitu.pl?t=html&p=downloads>`_)


Installation
-------------
$ pip install -e .

Acknowledgements
----------------
based on the implimentation by Amy Campbell from green lab (https://github.com/greenelab/staNMF) but with several important changes.
