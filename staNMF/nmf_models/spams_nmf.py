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
    n_components : int, optional with default None
        the number of patterns in NMF. If None, can be set later using
        self.n_components = <integer>.

    random_state : int, optional with default None
        the random seed used for determine initialization, None means that
        we use the default random seed, which will be different each time.

    bootstrap : bool, optional with default False
        Whether bootstrap the input matrix X

    verbose : bool, optional with default False
        whether to print out the intermediate results

    arguments : dict, optional with default {}
        the parameters used for spams package

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        the matrix of learned principle patterns, also known as dictionary

    n_components : int
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

    def __init__(self,
                 n_components=None,
                 random_state=None,
                 bootstrap=False,
                 verbose=False,
                 arguments={}):
        self.n_components = n_components
        self.random_state = random_state
        self.bootstrap = bootstrap
        self.verbose = verbose
        # Set the parameters for spams
        param = {
            "numThreads": -1,
            # minibatch size
            "batchsize": 512,
            # Number of columns in solution
            "K": n_components,
            "lambda1": 0,
            # Number of iterations to go into this round of NMF
            "iter": 2000,
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
        self.arguments = param
        for p in self.arguments:
            if p in arguments:
                self.arguments[p] = arguments[p]

        self.n_iter_ = self.arguments['iter']  # record number of iterations

    def set_n_components(self, K):
        self.n_components = int(K)
        # int is crucial because when K is double, code will be breakdown.
        self.arguments['K'] = int(K)

    def fit(self, X, y=None):
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
        np.random.seed(self.random_state)

        # Create bootstrapped X
        if self.bootstrap:
            n_samples = X.shape[0]
            bootstrap_X = X[np.random.choice(
                n_samples,
                n_samples,
                replace=True
            )]
        else:
            bootstrap_X = X

        # Compute the initialization dictionary
        initialization = initialguess(bootstrap_X.T, self.n_components)

        # Use spams to compute the PPs
        Dsolution = spams.trainDL(
                # Data matrix
                # we flip X because spams requires features as rows
                np.asfortranarray(bootstrap_X.T),
                # Initial guess as provided by initialguess()
                D=initialization,
                **self.arguments)
        self.components_ = Dsolution.T
        return self

    def transform(self, X):
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
            X=np.asfortranarray(X.T),
            # dict
            D=self.components_.T,
            # pos
            pos=True,
            lambda1=0,
            lambda2=0,
        )
        return coefs.toarray().T
