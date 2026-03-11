import numpy as np
from sklearn.metrics import mean_squared_error, r2_score # type: ignore	
from sklearn.linear_model import Ridge # type: ignore
from sklearn.preprocessing import PolynomialFeatures # type: ignore

# ---------------------------------------------------------------------
def harmt_polq(X, y, alpha):
    '''
    implements the regression with Ridge regularization
    '''
    hp = Ridge(alpha, fit_intercept=True)
    hp.fit(X, y)
    y_pred = hp.predict(X)

    mse = mean_squared_error(y, y_pred)
    R2 = hp.score(X, y)
    coeff = hp.coef_
    intercept = hp.intercept_
    
    return R2, coeff, intercept, mse, y_pred, hp

if __name__ == "__main__":
    None