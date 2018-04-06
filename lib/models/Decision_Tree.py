import numpy as np

class DecisionTree():

    def entropy(self, prob):
        '''
        Determines the entropy of a split.
        '''
        # replacing - values with 1 to avoid issues with np.log2 returning - inf
        # both 0 and 1 represent pure splits so the result will not change
        zero_mask = prob == 0
        prob[zero_mask] = 1

        log_2_prob = np.log2(prob)
        return -prob * log_2_prob

    def entropy_split(self, data, target, feature, debug=False):
		'''
		Determines which feature will result in the lowest entropy split.
		'''

        # splitting the data along a feature
        unique_class, class_counts = np.unique(data[:,feature], return_counts=True)

        weights = class_counts / target.shape[0]

		### REMOVE DEBUGGING STATEMENTS FROM FINAL VERSION
        if debug: print('Weights\n{}\n'.format(weights))

        # getting the split proportions
        splits = [target[data[:,feature] == uc] for uc in unique_class]

        prob = np.empty((unique_class.shape[0],2))

        for i, split in enumerate(splits):
            prob[i][0] = split.mean()
            prob[i][1] = 1 - split.mean()

        if debug: print('Probability\n{}\n'.format(prob))

        entropies = self.entropy(prob)

        if debug: print('entropies\n{}\n'.format(entropies))

        # transposing entropies shape so it can be multiplied with weights
        w_ent = entropies.T * weights

        return w_ent.sum()

    def find_best_split(self, data, target, split_candidates):
		'''
		finds the best split for a data set
		'''

        split_entropies = [self.entropy_split(data, target, split_candidate) for split_candidate in split_candidates]
        splt_ent = np.array(split_entropies)
        sort_ent_indx = splt_ent.argsort()
        best_ent_indx = sort_ent_indx[0]

        return split_candidates[best_ent_indx]

    def tree_builder(self, data, target, split_candidates=None):
		'''
		builds a decision tree
		'''

        if split_candidates is None:
            split_candidates = np.array([i for i in range(data.shape[1])])

        total_count = target.shape[0]
        true_count = target.sum()
        false_count = total_count - true_count

        if true_count == 0:
            return False
        elif false_count == 0:
            return True

        if split_candidates.shape[0] == 0:
            return true_count > false_count

        best_split = self.find_best_split(data, target, split_candidates)
        split_candidates = np.array([split_candidate for split_candidate in split_candidates if split_candidate != best_split])

        nodes = np.unique(data[:, best_split])
        subtree_dict = dict()

        for node in nodes:
            sub_mask = data[:, best_split] == node
            data_sub = data[sub_mask]
            tar_sub = target[sub_mask]

            subtree_dict[node] = self.tree_builder(data_sub, tar_sub, split_candidates)

        subtree_dict[None] = true_count > false_count

        return(self.features[best_split], subtree_dict)

    def fit(self, data, target):
		'''
		fits a decision tree to a data set.
		'''

        self.features = np.array(data.columns)

        data = np.array(data)
        target = np.array(target)

        self.tree = self.tree_builder(data, target)

    def predict(self, tree, element):
		'''
		predicts values for a single element in a data set
		'''

        if tree in [True, False]:
            return tree

        node, branches = tree
        branch = element[node]

        if branch not in branches:
            branch = None

        sub_tree = branches[branch]

        return self.predict(sub_tree, element)

    def score(self, data, target):
        '''
		determines the accuracy of the models predictions
		'''

		# gets predictions for every observations in the data
        self.tree_predictions = [self.predict(self.tree, data.iloc[i,:]) for i in range(data.shape[0])]

        true_count = 0

        for pred, actual in zip(self.tree_predictions, target):
            if pred == actual:
                true_count += 1
        return true_count / len(self.tree_predictions)
