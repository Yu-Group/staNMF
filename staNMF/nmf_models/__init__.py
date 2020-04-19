from .sklearn_nmf import sklearn_nmf
try:
    from .spams_nmf import spams_nmf
except Exception as e:
    print(e)
    print("spams package might not be avail.")
