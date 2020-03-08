import spams
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


def initialguess(X, K):
    '''
    Randomly samples K columns from X; sets input matrix guess to be a
    fortran array, and sets 'guesslist', a list of the column indices
    sampled from X for this replication of NMF;

    Parameters
    ----------
    X : numpy 2d-array
        full matrix

    K : int
        number of columns to select at random to be used as the 'initial
        guess' for the K PPs in spams NMF.

    Returns
    -------
    X[:, indexlist] : the initialization matrix of the PPs
    '''
    indexlist = np.random.choice(
                    np.arange(1, X.shape[1]),
                    K,
                    replace=False,
                )
    return X[:, indexlist]


class spams_nmf(BaseEstimator, TransformerMixin):
    '''
    Train non-negative matrix factorization via spams package

    Parameteres
    -----------
    n_components : int
        the number of patterns in NMF

    seed : int, optional with default None
        the random seed used for determine initialization, None means that
        we use the default random seed, which will be different each time.

    verbose : bool, optional with default False
        whether to print out the intermediate results

    Attributes
    ----------
    tag : str, static
        a tag of the class name

    components_ : array, shape (n_components, n_features)
        the matrix of learned principle patterns, also known as dictionary

    n_components_ : int
        the number of components

    n_iter_ : int
        the number of iterations in spams nmf

    Examples
    --------
    TODO : add an example

    References
    ----------
    TODO : add refs to spams package

    '''
    tag = 'spams_nmf'

    def __init__(self,
                 n_components,
                 seed=None,
                 verbose=False):
        self.n_components_ = n_components
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y=None, **kwargs):
        ''' Fit a NMF model using the spams package

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data matrix to be fitted by the model

        y : ignored

        Returns
        -------
        self
        '''
        # Set the seed for numpy.random
        np.random.set_state(self.seed)

        # Set the parameters for spams
        param = {
            "numThreads": -1,
            # minibatch size
            "batchsize": min(1024, X.shape[0]),
            # Number of columns in solution
            "K": int(self.n_components_),
            "lambda1": 0,
            # Number of iterations to go into this round of NMF
            "iter": 500,
            # Specify optimization problem to solve
            "mode": 2,
            # Specify convex set
            "modeD": 0,
            # Positivity constraint on coefficients
            "posAlpha": True,
            # Positivity constraint on solution
            "posD": True,
            # Limited information about progress
            "verbose": False,
            "gamma1": 0,
        }

        for p in param:
            if p not in kwargs:
                kwargs[p] = param[p]
        self.n_iter_ = kwargs['iter']  # record the number of iterations

        # Compute the initialization dictionary
        initialization = initialguess(X, self.K)

        # Use spams to compute the PPs
        Dsolution = spams.trainDL(
                # Matrix
                np.asfortranarray(X),
                # Initial guess as provided by initialguess()
                D=initialization,
                **kwargs)
        self.components_ = Dsolution
        return self

    def transform(self, X, **kwargs):
        '''
        Compute the loadings of X using the learned components

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            data matrix

        Returns
        -------
        coefs : array, shape (n_samples, n_components)
            transformed data
        '''
        coefs = spams.lasso(
            # data
            X=np.asfortranarray(X),
            # dict
            D=self.components_,
            # pos
            pos=True,
            lambda1=0,
            lambda2=0,
        )
        return coefs.toarray()
