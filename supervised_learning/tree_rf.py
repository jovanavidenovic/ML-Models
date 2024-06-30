import math
import random
import numpy as np
import pandas as pd

def all_columns(X, rand):
    """
    Function to return all columns of the dataset.
    Args:
        X: np.array, the dataset
        rand: random.Random, the random number generator    
    Returns:
        range: range, the range of values from 0 to the number of columns
    """
    return range(X.shape[1])

def random_sqrt_columns(X, rand):
    """
    Function to return a random subset of the columns of the dataset.
    Subset is of size sqrt(n), where n is the number of columns.
    Args:
        X: np.array, the dataset
        rand: random.Random, the random number generator
    Returns:
        c: list, the list of the indices of the columns
    """
    n = X.shape[1]
    c = rand.sample(range(n), int(np.sqrt(n)))
    return c

def get_bootstrap_sample(n, rand):
    """
    Function to return the indices of the in-bag and out-of-bag samples.
    Args:
        n: int, the number of samples
        rand: random.Random, the random number generator
    Returns:
        ib_idx: list, the list of the indices of the in-bag samples
        oob_idx: list, the list of the indices of the out-of-bag samples
    """
    ib_idx = rand.choices(population=range(n), k=n)
    oob_idx = [idx for idx in range(n) if idx not in ib_idx]
    return ib_idx, oob_idx

def misclassification_score(y_true, y_pred):
    """
    Function to compute the misclassification score.
    Misclassification score is the fraction of misclassified samples.
    Args:
        y_true: np.array, the true labels
        y_pred: np.array, the predicted labels
    Returns:
        score: float, the misclassification score
    """
    return len(y_true[y_true != y_pred]) / len(y_true)

class Tree:
    """
    Class to build a decision tree.
    """

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        """
        Constructor for the Tree class.
        Args:
            rand: random.Random, the random number generator
            get_candidate_columns: function, the function to get the candidate columns for splitting
            min_samples: int, the required minimum number of samples in a node to perform a split
        """
        self.rand = rand
        self.get_candidate_columns = get_candidate_columns
        self.min_samples = min_samples

    def __gini_impurity(self, y):
        """
        Function to compute the Gini impurity.
        Args:
            y: np.array, the labels     
        Returns:    
            gini: float, the Gini impurity  
        """
        if len(y) == 0:
            # if the node is empty, the Gini impurity is 0
            return 0
        
        # compute the probability of each class
        probs = np.bincount(y) / len(y)
        gini = 1 - np.sum(np.power(probs, 2))
        return gini
    
    def __choose_split(self, X, y):
        """
        Function to choose the best split for the node.
        If no split is found, the function returns None, None.
        Args:
            X: np.array, the dataset
            y: np.array, the labels
        Returns:
            best_feat: int, the index of the best feature to split or None if no split is found
            best_thr: float, the best threshold to split or None if no split is found
        """
        min_impurity = 1
        best_feat, best_thr = None, None

        candidate_cols = self.get_candidate_columns(X, self.rand)
        for feat in candidate_cols:
            feature_vals = X[:, feat]
            split_vals = np.unique(feature_vals)
            if len(split_vals) > 1:
                # if there is only one unique value, there is no split for this feature
                # take split values as the average of consecutive values
                split_vals = (split_vals[1:] + split_vals[:-1]) / 2    
                for split_val in split_vals:
                    left_node = feature_vals <= split_val
                    right_node = feature_vals > split_val
                    left_impurity = self.__gini_impurity(y[left_node])
                    right_impurity = self.__gini_impurity(y[right_node])
                    impurity = (len(y[left_node]) * left_impurity + len(y[right_node]) * right_impurity) / len(y)
                    
                    if impurity < min_impurity:
                        min_impurity = impurity
                        best_feat = feat
                        best_thr = split_val
                        if min_impurity == 0:
                            # if the impurity is 0, stop the search
                            return best_feat, best_thr
        
        return best_feat, best_thr

    def build(self, X, y):
        """
        Function to build a decision tree using CART algorithm.
        Args:
            X: np.array, the dataset
            y: np.array, the labels
        Returns:
            current_node: TreeNode, the (current) root node of the decision tree
        """
        zero_count = np.count_nonzero(y == 0)
        one_count = np.count_nonzero(y == 1)
        majority_class = 0 if zero_count >= one_count else 1

        if len(y) < self.min_samples:
            # if the node has less than min_samples, return a leaf node with the majority class
            return TreeNode(majority_class, None, None, None, None, self.rand)
        
        if np.unique(y).shape[0] == 1:
            # there is only one class in the node
            return TreeNode(majority_class, None, None, None, None, self.rand)
    
        split_feat, split_thr = self.__choose_split(X, y)
        if split_feat is None:
            # if there is no split found, return a leaf node with the majority class
            return TreeNode(majority_class, None, None, None, None, self.rand)    
        
        # get the indices of the samples in the left and right child nodes
        left_node = X[:, split_feat] <= split_thr
        right_node = X[:, split_feat] > split_thr

        # recursively build the left and right child nodes
        left_tree = self.build(X[left_node], y[left_node])
        right_tree = self.build(X[right_node], y[right_node])
                
        current_node = TreeNode(majority_class, split_feat, split_thr, left_tree, right_tree, self.rand)
        return current_node
    
class TreeNode:
    """
    TreeNode class to represent a node in the decision tree.
    """
    
    def __init__(self, majority_class, split_feat, split_thr, left_node, right_node, rand):
        """
        Constructor for the TreeNode class.
        Args:
            majority_class: int, the majority class in the node
            split_feat: int, the index of the feature to split
            split_thr: float, the threshold to split
            left_node: TreeNode, the left child node
            right_node: TreeNode, the right child node
            rand: random.Random, the random number generator
        """
        self.majority_class = majority_class
        self.split_feat = split_feat
        self.split_thr = split_thr
        self.left_node = left_node
        self.right_node = right_node
        self.rand = rand

    def predict(self, X):
        """
        Function to predict the labels of the samples.
        Args:
            X: np.array, the dataset
        Returns:
            y_pred: np.array, the predicted labels
        """
        y_pred = []
        for x in X:
            y_pred.append(self.__predict(x))             
        return np.array(y_pred)

    def __predict(self, x):
        """
        Function to predict the label of a sample.
        Args:
            x: np.array, the sample
        Returns:
            y_pred: int, the predicted label
        """
        if self.left_node is None and self.right_node is None:
            return self.majority_class
        
        if x[self.split_feat] <= self.split_thr:
            return self.left_node.__predict(x)
        else:
            return self.right_node.__predict(x)

    def importance(self, X_oob, y_oob):
        """
        Function to compute the importance of the features.
        Args:
            X_oob: np.array, the out-of-bag samples
            y_oob: np.array, the out-of-bag labels
        Returns:
            imps: np.array, the importance of the features
        """
        imps = np.zeros(X_oob.shape[1])
        # non-permuted score
        y_pred_orig = self.predict(X_oob)
        e_orig = misclassification_score(y_true=y_oob, y_pred=y_pred_orig)

        # permute each feature
        for feat_idx in range(X_oob.shape[1]):
            X_perm = X_oob.copy()
            self.rand.shuffle(X_perm[:, feat_idx])
            y_pred_perm = self.predict(X_perm)

            e_perm = misclassification_score(y_true=y_oob, y_pred=y_pred_perm)
            # measure error increase
            imps[feat_idx] = e_perm - e_orig
        return imps

class RandomForest:
    """
    RandomForest class to build a random forest.
    """
    def __init__(self, rand=None, n=50):
        """
        Constructor for the RandomForest class.
        Args:
            rand: random.Random, the random number generator
            n: int, the number of trees in the random forest
        """
        self.n = n
        self.rand = rand

    def build(self, X, y):
        """
        Function to build a random forest.
        Args:
            X: np.array, the dataset
            y: np.array, the labels
        Returns:
            RFModel: the random forest model
        """
        ib_indices = []
        oob_indices = []
        trees_built = []
        for _ in range(self.n):
            dec_tree = Tree(self.rand, get_candidate_columns=random_sqrt_columns)
            ib_idx, oob_idx = get_bootstrap_sample(len(X), self.rand)
            X_bs, y_bs = X[ib_idx], y[ib_idx]
            trees_built.append(dec_tree.build(X_bs, y_bs))
            ib_indices.append(ib_idx)
            oob_indices.append(oob_idx)

        return RFModel(trees_built, X, y, ib_indices, oob_indices)

class RFModel:
    """
    RFModel class to represent a random forest model.
    """

    def __init__(self, trees_built, X_train, y_train, ib_indices, oob_indices):
        """
        Constructor for the RFModel class.
        Args:
            trees_built: list, the list of decision trees in the random forest
            X: np.array, the dataset - train data
            y: np.array, the labels - train data
            ib_indices: list, the list of in-bag indices
            oob_indices: list, the list of out-of-bag indices
        """
        self.trees_built = trees_built
        self.X_train = X_train
        self.y_train = y_train
        self.ib_indices = ib_indices
        self.oob_indices = oob_indices

    def predict(self, X):
        """
        Function to predict the labels of the samples.
        Args:
            X: np.array, the dataset
        Returns:
            y_pred: np.array, the predicted labels
        """
        y_pred = []
        for dec_tree in self.trees_built:
            y_pred.append(dec_tree.predict(X))
        y_pred = np.array(y_pred)
        y_pred = [np.bincount(y_pred[:, i]).argmax() for i in range(y_pred.shape[1])]
        return y_pred

    def importance(self):
        """
        Function to compute the importance of the features.
        Returns:
            imps: np.array, the importance of the features
        """
        imps = np.zeros(self.X_train.shape[1])
        num_nonempty_oobs = 0
        for idx in range(len(self.trees_built)):
            dec_tree = self.trees_built[idx]
            oob_idx = self.oob_indices[idx]
            if len(oob_idx) != 0:
                X_oob = self.X_train[oob_idx]
                y_oob = self.y_train[oob_idx]

                imps += dec_tree.importance(X_oob, y_oob)
                num_nonempty_oobs += 1

        if num_nonempty_oobs != 0:
            imps /= num_nonempty_oobs
        return imps

def hw_tree_full(learn, test):
    """
    Function to build a decision tree and compute the misclassification score on learn and test set.
    Args:
        learn: tuple, the learn dataset and labels
        test: tuple, the test dataset and labels
    Returns:
        (train_mc, train_se): tuple, the misclassification score and standard error on the learn set
        (test_mc, test_se): tuple, the misclassification score and standard error on the test set
    """
    rand = random.Random(1)
    decision_tree = Tree(rand)

    X_train, y_train = learn
    X_test, y_test = test

    dec_tree = decision_tree.build(X_train, y_train)
    preds_train  = dec_tree.predict(X_train)
    preds_test = dec_tree.predict(X_test)

    train_mc = misclassification_score(y_true=y_train, y_pred=preds_train)
    test_mc = misclassification_score(y_true=y_test, y_pred=preds_test)

    train_se = math.sqrt(train_mc * (1 - train_mc) / len(y_train))
    test_se = math.sqrt(test_mc * (1 - test_mc) / len(y_test))

    return (train_mc, train_se), (test_mc, test_se)

def hw_randomforests(learn, test, N=100):
    """
    Function to build a random forest and compute the misclassification score on learn and test set.
    Args:
        learn: tuple, the learn dataset and labels
        test: tuple, the test dataset and labels
        N: int, the number of trees in the random forest
    Returns:
        (train_mc, train_se): tuple, the misclassification score and standard error on the learn set
        (test_mc, test_se): tuple, the misclassification score and standard error on the test set
    """
    rand = random.Random(1)

    rf = RandomForest(rand, N)
    X_train, y_train = learn
    X_test, y_test = test

    model = rf.build(X_train, y_train)

    preds_train = model.predict(X_train)
    preds_test = model.predict(X_test)
    
    train_mc = misclassification_score(y_true=y_train, y_pred=preds_train)
    test_mc = misclassification_score(y_true=y_test, y_pred=preds_test)

    train_se = math.sqrt(train_mc * (1 - train_mc) / len(y_train))
    test_se = math.sqrt(test_mc * (1 - test_mc) / len(y_test))

    return (train_mc, train_se), (test_mc, test_se)

def hw_rf_importance(X, y, rand, feat_names=None):
    """
    Function to compute the importance of the features when using random forest model.
    Also, the function computes the count of the features in the root nodes in 100 non-random trees.
    Args:
        X: np.array, the dataset
        y: np.array, the labels
        rand: random.Random, the random number generator
        feat_names: dictionary, the mapping of the feature indices to the feature names
    Returns:
        imps: dictionary, the importance of the features
        root_feats_count: dictionary, the count of the features in the root nodes
    """
    # measure the importance of the features in RF using permutation feature importance
    rf_imps = RandomForest(rand, 100)
    imps_model = rf_imps.build(X, y)
    imps = imps_model.importance()
    imps = sorted(enumerate(imps), key=lambda x: x[1], reverse=True)
    
    # measure the occurances of features in the root nodes of 100 non-random trees
    root_features = []
    for _ in range(100):
        ib_idx, _ = get_bootstrap_sample(len(X), rand)
        tree = Tree(rand)
        tree_root = tree.build(X[ib_idx], y[ib_idx])
        root_features.append(tree_root.split_feat)
    root_features = np.array(root_features)    
    root_feats_count = {}
    for feat in root_features:
        if feat not in root_feats_count:
            root_feats_count[feat] = 1
        else:
            root_feats_count[feat] += 1    
 
    if feat_names is not None:
        imps = {feat_names[idx]: imp for idx, imp in imps}
        root_feats_count = {feat_names[idx]: count for idx, count in root_feats_count.items()}

    return imps, root_feats_count 
