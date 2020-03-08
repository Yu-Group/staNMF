import math
import os
import shutil
import sys
import warnings
import argparse
import collections
import sklearn.preprocessing
from timeit import default_timer as timer

import numpy as np
from joblib import load, dump
import pandas as pd
from scipy.stats import pearsonr
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

FILENAME = "staNMFDicts_"


def load_example():
    '''
    Loads full data matrix from WuExampleExpression.csv file into numpy array;
    weights column names by their number of replicate occurances

    Returns
    -------
    X : array, shape (n_samples, n_features)
        data matrix extracted from WuExampleExpression.csv

    Examples
    --------
    >>> X = load_example()

    '''

    workingmatrix = pd.read_csv('../Demo/WuExampleExpression.csv', index_col=0)

    # weight each column (gene) by 1 / its occurences in replicates
    colnames = workingmatrix.columns.values
    colnames = [(str(x).split('.'))[0] for x in colnames]
    colUnique = np.unique(colnames)
    colNum = np.zeros(len(colUnique))
    weight = np.zeros(len(colnames))

    for i in range(len(colUnique)):
        colNum[i] = list(colnames).count(colUnique[i])
        weight[i] = 1/(colNum[i])

    workingmatrix = workingmatrix.apply(
        lambda x: weight * x,
        axis=1,
    )
    workingmatrix = workingmatrix.applymap(lambda x: math.sqrt(x))

    X = (np.array(workingmatrix).astype(float)).T
    return X


def findcorrelation(self, A, B):
    '''
    Construct k by k matrix of Pearson product-moment correlation
    coefficients for every combination of two columns in A and B

    Parameters
    ----------
    A : array, shape (n_components, n_features)
        first NMF solution matrix

    B : array, shape (n_components, n_features)
        second NMF solution matrix, of same dimensions as A

    Returns
    -------
    X : array shape (n_components, n_components)
        array[a][b] is the correlation between column 'a' of X
        and column 'b'

    '''
    A_std = sklearn.preprocessing.scale(A)
    B_std = sklearn.preprocessing.scale(B)
    return A_std.T @ B_std / A.shape[0]


class staNMF:
    '''Python 3 implementation of Siqi Wu's 03/2016 Stability NMF (staNMF)

    Solves non-negative matrix factorization for a range of principal patterns
    (PPs) with either different initializations or bootstrapped samples.

    Parameters
    ----------
    X : numpy array, shape (n_samples, n_features)
        2d numpy array containing the data.

    folderID : str, optional with default ""
        allows user to specify a unique (to the user's working directory)
        identifier for the FILENAME folder that the runNMF method creates.

    K1 : int, optional with default 15
        lowest number of PP's (K) tested

    K2 : int, optional with default 30
        highest number of PP's (K) tested

    seed : int, optional with default 123
        set numpy random seed

    replicates : int or tuple of ints of length 2, optional with default
    int 100
        specify the bootstrapped repetitions to be performed on each value of K
        for use in stability analysis; if a list of length 2: self.replicates
            is set to a list of ints between the first and second elements of
            this tuple. If int: self.replicates is set to range(integer).

    NMF_finished : bool, optional with default False
        True if runNMF has been completed for the dataset. To surpass NMF step
        if fileID file already contains factorization solutions for X in your
        range [K1, K2], set to True.

    parallel : bool, optional with default False
        True if NMF is to be run in parallel such that the instability
        calculation should write a file for each K containing its instability
        index.

    '''

    def __init__(self, X, folderID="", K1=15, K2=30,
                 seed=123, replicates=100,
                 NMF_finished=False, parallel=False):
        warnings.filterwarnings("ignore")
        self.K1 = K1
        self.K2 = K2
        self.seed = seed
        self.guess = np.array([])
        self.guessdict = {}
        self.parallel = parallel
        if isinstance(replicates, int):
            self.replicates = range(replicates)
        elif isinstance(replicates, tuple):
            start, stop = replicates
            self.replicates = range(replicates[0], replicates[1])
        self.X = X
        self.folderID = folderID
        self.NMF_finished = NMF_finished
        self.instabilitydict = {}
        self.instability_std = {}
        self.instabilityarray = []
        self.instabilityarray_std = []
        self.stability_finished = False

    def runNMF(self, nmf_class, **kwargs):
        '''
        Iterate through range of integers between the K1 and K2 provided (By
        default, K1=15 and K2=30), run NMF using the model; output NMF matrix
        files (.csv form).

        Parameters
        ----------
        nmf_class : a class that can be initiated to fit NMF models

        Returns
        -------
        None

        Side effects
        ------------
        (k2-k1) folders, each containing files for every replicate
        (labeled nmf_model_<nmf_class>_<replicate>.joblib).

        Raises
        ------
        OSError
            the path cannot be created.

        '''

        self.NMF_finished = False
        numPatterns = np.arange(self.K1, self.K2+1)
        for k in range(len(numPatterns)):
            K = numPatterns[k]
            path = (
                "./" + FILENAME + self.folderID + "/K=" + str(K) + "/"
            )
            try:
                os.makedirs(path)
            except OSError:
                if not (os.path.isdir(path)):
                    raise
            m, n = np.shape(self.X)

            print("Working on " + str(K) + "...\n")

            # create an object from the nmf_class
            for l in self.replicates:
                nmf_model = nmf_class(
                    n_components=K,
                    seed=self.seed + 100 * l
                )
                nmf_model.fit(self.X, **kwargs)  # fit nmf model
                # write model to a joblib file in the path folder
                outputfilename = (
                    "nmf_model_" + nmf_class.tag + '_' + str(l) + ".joblib"
                )
                outputfilepath = os.path.join(path, outputfilename)
                dump(nmf_model, outputfilepath)

            self.NMF_finished = True

    def amariMaxError(self, correlation):
        '''
        Computes what Wu et al. (2016) described as a 'amari-type error'
        based on average distance between factorization solutions

        Return:
       Amari distance distM

        Arguments:
        :param: correlation: k by k matrix of pearson correlations

        Usage: Called by instability()
        '''

        n, m = correlation.shape
        maxCol = np.absolute(correlation).max(0)
        colTemp = np.mean((1-maxCol))
        maxRow = np.absolute(correlation).max(1)
        rowTemp = np.mean((1-maxRow))
        distM = (rowTemp + colTemp)/(2)

        return distM

    def instability(self, nmf_class, k1=0, k2=0):
        '''
        Performs instability calculation for NMF models for each K
        within the range entered

        Parameters
        ----------
        nmf_class : class
            the nmf class for which we compute the stability

        k1 : int, optional, default self.K1
            lower bound of K to compute stability

        k2 : int, optional, default self.K2
            upper bound of K to compute instability

        Returns
        -------
        None
        
        Side effects
        ------------
        "instability.csv" containing instability index
        for each K between and including k1 and k2; updates
        self.instabilitydict (required for makeplot())
        '''
        if k1 == 0:
            k1 = self.K1
        if k2 == 0:
            k2 = self.K2

        numReplicates = len(self.replicates)

        if self.NMF_finished is False:
            print("staNMF Error: runNMF is not complete\n")
        else:
            numPatterns = np.arange(k1, k2+1)
            n_features = self.X.shape[1]
            
            # loop through each number of PPs
            for k in numPatterns:
                print("Calculating instability for " + str(k))
                
                # load the dictionaries
                path = (
                    "./" + FILENAME + self.folderID + "/K=" + str(k)+"/"
                )
                Dhat = np.zeros((numReplicates, n_features, k))

                for replicate in range(numReplicates):
                    inputfilename = (
                        "nmf_model_" + nmf_class.tag 
                        + "_" + str(replicate) + ".joblib"
                    )
                    inputfilepath = os.path.join(path, inputfilename)
                    model = load(inputfilepath)
                    Dhat[replicate] = model.components_

                # compute the distance matrix between each pair of dicts
                distMat = np.zeros(shape=(numReplicates, numReplicates))

                for i in range(numReplicates):
                    for j in range(i, numReplicates):
                        x = Dhat[i]
                        y = Dhat[j]

                        CORR = findcorrelation(x, y)
                        distMat[i][j] = self.amariMaxError(CORR)
                        distMat[j][i] = distMat[i][j]

                # compute the instability and the standard deviation
                self.instabilitydict[k] = (
                    np.sum(distMat) / (numReplicates * (numReplicates-1))
                )
                # The standard deviation of the instability is tricky:
                # It is a U-statistic and in general hard to compute std.
                # Fortunately, there is a easy-to-understand upper bound.
                self.instability_std[k] = (
                    np.sum(distMat ** 2)
                    / (numReplicates * (numReplicates - 1))
                    - self.instabilitydict[k] ** 2
                ) ** .5 * (2 / distMat.shape[0]) ** .5
                
                # write the result into csv file
                outputfile = path + "instability.csv"
                pd.DataFrame({
                    'K': [k],
                    'instability': [self.instabilitydict[k]],
                    'instability_std': [self.instability_std[k]],
                }).to_csv(outputfile, mode='a', header=False, index=False)

    def get_instability(self):
        '''
        Retrieves instability values calculated in this instance of staNMF

        Returns:
        dictionary with keys K, values instability index

        Usage: Called by user (not required for output of 'instablity.csv', but
        returns usable python dictionary of these calculations)
        '''
        if self.stability_finished:
            return self.instabilitydict
        else:
            print("Instability has not yet been calculated for your NMF"
                  "results. Use staNMF.instability() to continue.")

    def plot(self, dataset_title="Drosophila Spatial Expression Data", xmax=0,
             xmin=-1, ymin=0, ymax=0, xlab="K", ylab="Instability Index"):
        '''
        Plots instability results for all K's between and including K1 and K2
        with K on the X axis and instability on the Y axis

        Parameters
        ----------

        dataset_title : str, optional, default "Drosophila
            Expression Data"
            The title used in the plot

        ymax : float, optional,  default
            largest Y + largest std(Y) * 2  + (largest Y/ # of points)
            the maximum y axis limit

        xmax : float, optional, default K2+1

        xlab : string, default "K"
            x-axis label

        ylab : string, default "Instability Index"
            y-axis label

        Returns
        -------
        None

        Side effects
        ------------
        A png file named <dataset_title>.png is saved.

        '''
        kArray = []
        self.instabilityarray = []
        self.instabilityarray_std = []
        for K in range(self.K1, self.K2+1):
            kpath = (
                "./" + FILENAME + "{}/K={}/instability.csv"
            ).format(self.folderID, K)
            df = pd.read_csv(kpath, header=None, index_col=False)
            kArray.append(int(df.iloc[0, 0]))
            self.instabilityarray.append(float(df.iloc[0, 1]))
            self.instabilityarray_std.append(float(df.iloc[0, 2]))
        if xmax == 0:
            xmax = self.K2 + 1
        if xmin == -1:
            xmin = self.K1 - .1
        if ymax == 0:
            ymax = max(self.instabilityarray) \
                   + max(self.instabilityarray_std) * 2 \
                   + (max(self.instabilityarray) / len(self.instabilityarray))
        plt.errorbar(x=kArray,
                     y=self.instabilityarray,
                     yerr=np.array(self.instabilityarray_std)*2)
        plt.axis([xmin, xmax, ymin, ymax])
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.axes.titlesize = 'smaller'
        plt.title(str('Stability NMF Results: Principal Patterns vs.'
                      'Instability in ' + dataset_title))
        plotname = str(dataset_title + ".png")
        plt.savefig(plotname)

    def ClearDirectory(self, k_list):
        '''
        A storage-saving option that clears the entire directory of each K
        requested, including the instability.csv file in each folder

        :param: k_list (list, required) list of K's to delete corresponding
        directories of

        NOTE: this should only be used after stability has been calculated for
        each K you wish to delete.
        '''
        for K in k_list:
            path = ("./" + FILENAME + "{}/K={}/").format(self.folderID, K)
            shutil.rmtree(path)
