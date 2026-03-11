import numpy as np

# ---------------------------------------------------------------------
def std_scaling(x):
    '''
    Standard scaling of feature 'q'
    '''
    x_std = np.std(x)
    x_mean = np.mean(x)

    x_resc = (x - x_mean) / x_std

    return x_resc

if __name__ == "__main__":
    None