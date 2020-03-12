from sklearn.decomposition import NMF
from sklearn.decomposition.nmf import _initialize_nmf
import numpy as np


class sklearn_nmf(NMF):
    '''
    Train non-negative matrix factorization via sklearn package

    Parameteres
    -----------
    bootstrap : bool, optional with default False
        Do bootstrap to X before fitting

    All the parameters in sklearn NMF

    Attributes
    ----------
    All the attributes in sklearn.decomposition.nmf.NMF

    Examples
    --------
    TODO : add an example

    References
    ----------
    TODO : add refs to spams package

    '''
    def __init__(self, n_components=None, init=None, solver='cd',
                 beta_loss='frobenius', tol=1e-4, max_iter=200,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 shuffle=False, bootstrap=False):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle
        self.bootstrap = bootstrap

    def set_n_components(self, K):
        self.n_components = K

    def fit(self, X, y=None, **kwargs):
        '''
        Fit NMF model using sklearn package

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data matrix to be fitted by the model

        y : ignored

        bootstrap : bool, optional with default False
            Whether bootstrap the input matrix X

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
                replace=True,
            )]
        else:
            bootstrap_X = X

        # Call the super fit method
        return super(sklearn_nmf, self).fit(bootstrap_X, **kwargs)
