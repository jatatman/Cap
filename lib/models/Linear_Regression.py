class Linear_Regression():
    '''
    Linear regression using ordinary least squares\
    to determine the distance between points.
    '''

    def __init__(self):
        pass


    def fit(self, X=None, y=None, reg=False, alpha=None, dg_fr=None):
        import numpy as np

        '''
        Creates a line of best fit for the data using linear regression.

        Parameters:

            X: Data
            y: target
            reg: regularization
            alpha:
            dg_fr: degrees of freedomm
        '''

        X, y = data_check(X,y)

        ones = np.ones(X.shape[0])
        X_ = np.column_stack((ones, X))


        if reg == False:
            self.betas = np.linalg.inv(X_.T.dot(X_)).dot(X_.T).dot(y)

            if alpha != None:
                print('In order to use the alpha parameter\
                regularization must be set to TRUE')

            if deg_freedom != None:
                print('In order to use the degrees of freedom parameter\
                 regularization must be set to TRUE')

        if reg == True:
            inverse = np.linalg.inv()
            self.betas = inverse(X_.T.dot(X_) + alpha * np.eye(deg).dot(X_.T).dot(y)

        return self.betas


    def score(self, X=None, y=None):
        '''
        Returns R2 score.
        '''

        y_pred = self.betas[0] + X.dot(self.betas[1:])

        RSS = ((y_pred - y) ** 2).sum()
        TSS = ((y - y.mean()) ** 2).sum()

        R2 = 1 - (RSS/TSS)

        return R2


    def data_check(X,y):

        '''
        Converts data to a numpy ndarray.
        '''

        Xc == X.copy
        yc == y.copy

        if type(Xc) != np.ndarray:
            try:
                Xc = np.array(Xc)
                print(type(Xc))
            except:
                raise Exception('Failed to convert data to np.ndarray')
        if type(yc) != np.ndarray:
            try:
                yc = np.array(yc)
                print(type(yc))
            except:
                raise Exception('Failed to convert target to np.ndarray')

        return Xc, yc
