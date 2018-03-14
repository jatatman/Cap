import numpy as np

def Class DecisionTree():

	def __init__(self):
		pass

	def entropy(self, prob):
    	'''

    	'''
    	# replacing - values with 1 to avoid issues with np.log2 returning - inf
    	# both 0 and 1 represent pure splits so this change will not effect the result

    	zero_mask = prob == 0
    	prob[zero_mask] = 1

    	log_2_prob = np.log2(prob)
    	return -prob * log_2_prob

    def entropy_split(self, data, target, feature, debug=False):

        # splitting data along a feature
        unique_class, class_counts = np.unique(data[:,feature], return_counts=True)

        weights = class_counts/ target.shape[0]

        if debug: print('Weights\n{}\n'.format(weights))
        # getting the split proportions
        splits = [target[data[:,feature] == uc] for uc in unique_class]

        prob = np.empty((unique_class.shape[0],2))

        for i, split in enumerate(splits):
            prob[i][0] = split.mean()
            prob[i][1] = 1 - split.mean()

        if debug: print('Probability\n{}\n'.format(prob))

        entropies = entropy(prob)

        if debug: print('entropies\n{}\n'.format(entropies))

            # transposing entropies shape so it can be multiplied with weights
        w_ent = entropies.T * weights

        return w_ent.sum()

	def find_best_split(data, target):

    	split_entropies = []

    	for i in range(data.shape[1]):
        	ent = entropy_split(data, target, i)
        	split_entropies.append(ent)

    	splt_ent = np.array(split_entropies)
    return splt_ent.argsort()

### from Data Science From Scratch

def build_tree_id3(inputs, split_candidates=None):

	# if this is oru first pass,
	# all keys of the first input are split split_candidates
	if split canddiates is None:
		split_candidates = inputs[0][0].keys()
	# count Trues and Falses in the inputs
	num_inputs = len(inputs)
	num_trues = len([label for item, label in inputs if label])
	num_falses = num_inputs - num_trues


	if num_trues == 0: return False # no Trues? return a "False" leaf
	if num_falses == 0: return True # no Falses? return a "True" leaf

	if not split_candidates: # if no split candidates left
		return num_trues >= num_falses # return majority leaf

	# otherwise split on the best attributes
	best_attribute = min(split_candidates,
	key=partial(partition_entropy_by, inputs))

	parititons = partition_by(inputs, best_attribute)
	new_canddidates = [a for a in split_candidates
	if a != best_attribute]

	# recursively build the subtrees
	subtrees = { attribute_value : build_tree_id3(subset, new_candidates)
	for attribute_value, subset in partition.iteritems()}

	subtreees]None = num_trues > num_falses # default case
