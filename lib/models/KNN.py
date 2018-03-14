import numpy as np

class KNN():

    def __init__(self, neighbors=1, distance_metric='minkowski', p=2):
        ''' K-Nearest_Neighbors Classifier'''

        self.n = neighbors
        self.metric = distance_metric
        self.p = p

    def fit(self, X, y):

        self.X_fit, self.y_fit = self.data_check(X,y)


    def find_distance(self, X):

        # remove euclidian and just pass p=2 to minkowski to get euclidian calculation
        # if there is no performance difference

        if self.metric == 'manhattan':

            vector_dist = [(np.absolute(self.X_fit - X[i])) for i in range(X.shape[0])]
            distance = [[x.sum() for x in X] for X in vector_dist]

        elif self.metric == 'minkowski':

            vector_dist = [(np.absolute(self.X_fit - X[i])**self.p) for i in range(X.shape[0])]
            distance = [[(x.sum())**(1/self.p) for x in X] for X in vector_dist]

        return np.array(distance)

    def prediction(self,X,y):

        distance = self.find_distance(X)

        min_dist_index = [distance[i].argsort()[0:self.n] for i in range(distance.shape[0])]
        nbr_trg = [self.y_fit[min_dist_index[i]] for i in range(len(min_dist_index))]

        pred =[]
        for i in range(len(nbr_trg)):
            trg_id, trg_freq = np.unique(nbr_trg[i], return_counts=True)
            pred_index = trg_freq.argsort()[-1]
            pred.append(trg_id[pred_index])

        pred = np.array(pred)
        return pred
#     return pred, trg_freq, trg_id

    def score(self, X,y):

        X,y = self.data_check(X,y)

        correct = 0
        incorrect = 0

        pred = self.prediction(X,y)

        for i in range(y.shape[0]):
            if y[i] == pred[i]:
                correct += 1
            else:
                incorrect += 1

        return (correct / (correct + incorrect)) * 100

    def data_check(self, X,y):
        '''
        Converts data to a numpy ndarray.
        '''

        if type(X) != np.ndarray:
            try:
                X = np.array(X)

            except:
                raise Exception('Failed to convert data to np.ndarray')
        if type(y) != np.ndarray:
            try:
                y = np.array(y)

            except:
                raise Exception('Failed to convert target to np.ndarray')

        return X, y
