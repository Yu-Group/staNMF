import torch as th
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau

# define model autoencoder
class Encoder(th.nn.Module):
    '''
    Encoder class that has only one hidden layer.

    Parameters
    ----------
    input_dim : int, required
        The dimension of the input signal
    num_neurons1 : int, required
        #neurons in the first layer

    num_neurons2 : int, required
        #neurons in the second layer

    negative_slope: float, optional with default 0.01
        the slope for the leaky RELU

    '''
    def __init__(
        self,
        input_dim,
        num_neurons1,
        num_neurons2,
        negative_slope=0.01,
    ):
        super(Encoder, self).__init__()
        self.pipe = th.nn.Sequential(
            th.nn.Linear(  # Linear Layer
                input_dim,
                num_neurons1,
                bias=False
            ),
            th.nn.LeakyReLU(  # Leaky Activation
                negative_slope=negative_slope,
                inplace=True
            ),
            th.nn.Linear(  # Linear Layer
                num_neurons1,
                num_neurons2,
                bias=False
            ),
        )

    def forward(self, x):
        return th.abs(self.pipe(x))  # output is non-negative.


class Decoder(th.nn.Module):
    '''
    Decoder class that's linear

    Parameters
    ----------
    num_neurons2 : int, required
        the dimension of the input layer

    output_dim : int, required
        the dimension of the output dimension

    init_patterns : array of shape (n_in_features, n_out_features), optional
        The initial patterns/weights used in the linear layer. Default None.

    random_init_max : float, optional with default 1e-4
        The maximal weight value when generating the initial_patterns randomly
        Only useful when init_patterns is None.
    '''
    def __init__(
        self,
        num_neurons2,
        output_dim,
        init_patterns=None,
        random_init_max=1e-4,
    ):
        super(Decoder, self).__init__()
        if init_patterns is None:
            init_patterns = np.random.uniform(
                0,
                random_init_max,
                (num_neurons2, output_dim)
            ).astype(np.float32)
        self.patterns = th.nn.Parameter(
            th.tensor(init_patterns),
            requires_grad=True,
        )

    def forward(self, x):
        return th.mm(
            x,
            th.abs(self.patterns),
        )


class mydataset(data.Dataset):
    def __init__(self, X):
            self.X = th.FloatTensor(X.astype('float'))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
            return self.X[index]


# define NMF sklearn-type model
class torch_nmf(BaseEstimator):
    def __init__(
        self,
        n_features,
        n_neurons1=400,
        bootstrap=True,
        n_components=None,
        max_epochs=1000,
        batch_size=1000,
        random_init_max=1e-4,
        init_patterns=None,
        beta1=0.9,
        beta2=0.999,
        cpu_workers=4,
        shuffle=True,
        learning_rate=0.01,
        device='auto',
        random_state=None,
        record=True,
        use_scheduler=True,
        scheduler_patience=10,
    ):
        self.n_features = n_features
        self.n_neurons1 = n_neurons1
        self.n_components = n_components
        self.max_epochs = max_epochs
        self.bootstrap = bootstrap
        self.cpu_workers = cpu_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.init_patterns = init_patterns
        self.random_init_max = random_init_max
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.random_state = random_state
        self.record = record
        self.use_scheduler = use_scheduler
        self.scheduler_patience = scheduler_patience
        if device == 'auto':
            use_cuda = th.cuda.is_available()
            self.device = th.device("cuda:0" if use_cuda else "cpu")
        else:
            self.device = device


    def set_n_components(self, K):
        self.n_components = K

    @property
    def components_(self):
        try:
            patterns = self.decoder_.patterns.abs().to('cpu')
            normed_patterns = patterns / patterns.norm(dim=1)[:, None]
            return normed_patterns.data.numpy()
        except:
            return None

    def fit_transform(self, X, y=None):
        # Set the seed for numpy.random
        np.random.seed(self.random_state)

        if self.bootstrap:
            n_samples = X.shape[0]
            bootstrap_X = X[np.random.choice(
                n_samples,
                n_samples,
                replace=True,
            )]
        else:
            bootstrap_X = X

        training_set = mydataset(bootstrap_X)
        train_loader = data.DataLoader(
            training_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.cpu_workers,
        )
        if self.record:
            self.obj_list_ = []

        self.encoder_ = Encoder(
            self.n_features, 
            num_neurons1=self.n_neurons1,
            num_neurons2=self.n_components,
        )
        self.decoder_ = Decoder(
            self.n_components,
            self.n_features,
            random_init_max=self.random_init_max,
            init_patterns=self.init_patterns,
            )
        self.optim_ = th.optim.Adam(
            params=(list(self.encoder_.parameters()) 
                    + list(self.decoder_.parameters())),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
        )
        self.loss_ = th.nn.MSELoss(reduction='mean')
        if self.use_scheduler:
            self.scheduler_ = ReduceLROnPlateau(
                self.optim_,
                patience=self.scheduler_patience)
        for epoch in range(self.max_epochs):
            for batch in train_loader:
                self.encoder_.zero_grad()  # zero out the gradient
                self.decoder_.zero_grad()
                batch = batch.to(self.device)
                coefs = self.encoder_(batch)
                preds = self.decoder_(coefs)
                obj = self.loss_(preds, batch)
                obj.backward()
                self.optim_.step()
            if self.use_scheduler:
                preds = self.decoder_(self.encoder_(training_set.X))
                obj = self.loss_(preds, training_set.X)
                self.scheduler_.step(obj)
            if self.record:
                preds = self.decoder_(self.encoder_(training_set.X))
                obj = self.loss_(preds, training_set.X)
                self.obj_list_.append(np.asscalar(obj.data.numpy()))

        return self.decoder_(self.encoder_(
            th.FloatTensor(X.astype('float'))
        )).data.numpy()

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self
        
    def transform(self, X):
        training_set = mydataset(X)
        return self.decoder_(self.encoder_(training_set.X)).data.numpy()
