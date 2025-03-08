import numpy as np

class DecisionTree:
    class Node:
        def __init__(self, feature=None, threshold=None, l_child=None, r_child=None, dtype=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.l_child = l_child
            self.r_child = r_child
            self.dtype = dtype
            self.value = value

        def is_leaf_node(self):
            return self.value is not None

    def __init__(self, max_depth=10, min_samples=5, imbalance_threshold=0.90):
        self.max_depth_val = max_depth
        self.min_samples = min_samples
        self.imbalance_threshold = imbalance_threshold
        self.root = None

    def _grow_tree(self, data: np.ndarray, features: list, depth: int):

        if self._is_homogeneous(data): #basecase
            return self.Node(value=self.majority_class(data))

        return ...

    def predict(self):
        pass

    def _predict_sample(self):
        pass

    def _grow_tree(self):
        pass

    def _majority_class(self, data: np.ndarray):
        labels, counts = np.unique(data[:, -1], return_counts=True)
        return labels[np.argmax(counts)]

    def _is_homogeneous(self, data: np.ndarray):
        _, cls_counts = np.unique(data[:, -1], return_counts=True)
        return len(cls_counts) == 1  #true if data only consists of 1 class
    
    def _max_depth(self, depth: int):
        return depth >= self.max_depth_val

    def _is_imbalanced(self, data: np.ndarray):
        _, counts = np.unique(data[:, -1], return_counts=True)
        purity = max(counts) / np.sum(counts)
        return purity > self.imbalance_threshold
    
    def _small_node_size(self, data: np.ndarray):
        return len(data) <= self.min_samples

    def _impurity_measure_numeric(self):
        pass

    def _impurity_measure_categorical(self):
        pass

    def _gini_index(self):
        pass

    def _split_data(self):
        pass
