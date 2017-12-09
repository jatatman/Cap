import numpy as np 


def StandardScaler(data):

	return (data.values - data.mean().values) / (data.max().values - data.min().values)

class LinearRegression():

	betas = 0


	def fit(X=None, y=None):
    
    	X = X - X.mean()
    	y = y - y.mean()
    
    	betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    	return betas

	def score(X=None, y=None):
    
    	y_pred = betas[0] + X.dot(betas)
    
    	RSS = ((y_pred - y) ** 2).sum()
    	TSS = ((y - y.mean()) ** 2).sum()
    
    	R2 = 1 - (RSS/TSS)

    	return(R2)

class LogisticRegression():



	def fit(X, y):
    X = X - X.mean()
    y = y - y.mean()
    
    betas = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    
    return betas


	def predict(X=None, y=None):
		e = np.exp(1)

		numerator = e ** (betas[0] + betas * X)
		denomenator = e ** (betas[0] + betas * X) + 1




class KNN():

	# the model fit is the data

	def predict(X, neighbors=1):
    
    pred_list = []
    # add exception for when neighbors > len(X)
    for i in range(len(y_ts))

    	# getting distance
        dist = np.sqrt((X_tr[i][0] - X[:,0])**2 + (X_tr[i][1] - X[:,1])**2 )

        min_dist_index = dist.argsort()[0:neighbors]
        trgt_of_nbr = y_ts[min_dist_index]
    
 
        trgt_freq, trgt_id = np.unique(trgt_of_nbr, return_counts=True)
        pred_index = trgt_freq.argsort()[0]
        prediction = trgt_freq[pred_index]
        
        pred_list.append
        
    return pred_list

