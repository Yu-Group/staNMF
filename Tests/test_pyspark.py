import staNMF
from staNMF.nmf_models import sklearn_nmf
import numpy as np
if __name__ == '__main__':
    X = np.abs(np.random.normal(0, 1, (100,20)))
    sta_nmf = staNMF.staNMF(
        X=X,
        K1=2,
        K2=4,
        replicates=3,
        parallel_mode='pyspark',
    )
    sta_nmf.runNMF(sklearn_nmf(bootstrap=True))
    sta_nmf.instability('sklearn_nmf')
    sta_nmf.plot()
    print("success.")
