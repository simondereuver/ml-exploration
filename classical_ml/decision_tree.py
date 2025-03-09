import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

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
            return self.Node(value=self._majority_class(data))

        feature, threshold = self._feature_to_split(data, features)
        if feature is None or threshold is None:
            return self.Node(value=self._majority_class(data))
        
        if feature is None or threshold is None: #no better gini was found
            return self.Node(value=self._majority_class(data))

        l_subset, r_subset = self.__split_data(data, feature, threshold)

        if l_subset.size > 0:
            l_child = self._grow_tree(l_subset, features, depth=depth + 1) #add dtype and use dtype in predict
        else:
            l_child = None
        if r_subset.size > 0:
            r_child = self._grow_tree(r_subset, features, depth=depth + 1)
        else:
            r_child = None
        return self.Node(feature=feature, threshold=threshold, l_child=l_child, r_child=r_child)

    def __split_data(self, data: np.ndarray, feature : list, threshold):
        if np.issubdtype(data[:,feature].dtype, np.number): #numeric feature
            mask = (data[:,feature]) < threshold
            subset1 = data[mask]
            subset2 = data[~mask]
        else: #categorical feature
            mask = (data[:,feature]) == threshold
            subset1 = data[mask]
            subset2 = data[~mask]
        return subset1, subset2

    def fit(self, data: np.ndarray, labels: np.ndarray, features: list = None):
        if features is None:
            features = list(range(data.shape[1]))
        data = np.hstack((data, labels.reshape(-1, 1)))
        self.root = self._grow_tree(data, features, depth=1)

    def predict(self, data: np.ndarray):
        predictions = []
        for data_row in data:
            predictions.append(self.__predict_with_tree(self.root, data_row))
        return np.array(predictions)

    def _predict_with_tree(self, node, data_point):
        if node.is_leaf_node(): #basecase
            return node.value
        #print(type(data_point[tree.feature]))
        if isinstance(data_point[node.feature], (int, float, np.number)):
            if data_point[node.feature] < node.threshold:
                return  self.__predict_with_tree(node.l_child, data_point)
            else:
                return  self.__predict_with_tree(node.r_child, data_point)
        else:
            if data_point[node.feature] == node.threshold:
                return self.__predict_with_tree(node.l_child, data_point)
            else:
                return self.__predict_with_tree(node.r_child, data_point)

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

    def _impurity_measure_numeric(self, data_split: np.ndarray) -> list[np.float64, np.float64]:
        best_threshold = None
        min_gini = 1
        n_f_vals = data_split.shape[0]
        
        #naive way of handling this right now, as we need to sort each time, could be improved
        sorted_f_indices = np.argsort(data_split[:, 0]) #sort based on feature values
        sorted_f_vals = data_split[sorted_f_indices]
        for i in range(n_f_vals - 1): #-1 to stop early for edgecase of last index
            curr_f_val = sorted_f_vals[i, 0]
            next_f_val = sorted_f_vals[i + 1, 0]
            curr_cls_val = sorted_f_vals[i, 1]
            next_cls_val = sorted_f_vals[i + 1, 1] 

            if curr_f_val == next_f_val: #skip to next iteration as this puts both f_vals on same side anyways
                continue
            if curr_cls_val == next_cls_val: #skip to next iteration as class label doesnt change
                continue

            threshold = (curr_f_val + next_f_val) / 2.0
            mask = sorted_f_vals[:, 0] < threshold #split based on threshold
            l_side = sorted_f_vals[mask]
            r_side = sorted_f_vals[~mask]

            if l_side.shape[0] == 0 or r_side.shape[0] == 0:
                continue

            l_weight = l_side.shape[0] / n_f_vals
            r_weight = r_side.shape[0] / n_f_vals
            
            _, l_counts = np.unique(l_side[:,-1], return_counts=True)
            _, r_counts = np.unique(r_side[:,-1], return_counts=True)

            l_gini = self._gini_index(l_counts)
            r_gini = self._gini_index(r_counts)
            curr_gini = (l_weight * l_gini) + (r_weight * r_gini)
            if curr_gini < min_gini:
                min_gini = curr_gini
                best_threshold = threshold

        return (min_gini, best_threshold) if best_threshold is not None else (None, None) #no valid split in this feature

    def _impurity_measure_categorical(self, data_split: np.ndarray):
        min_gini = 1
        n_vals = data_split.shape[0]
        unique_vals = np.unique(data_split[:, 0])
        best_split_value = None
        
        for val in unique_vals:
            mask = data_split[:, 0] != val
            left_side = data_split[mask]
            right_side = data_split[~mask]
            
            if left_side.shape[0] == 0 or right_side.shape[0] == 0:
                continue
            
            left_weight = left_side.shape[0] / n_vals
            right_weight = right_side.shape[0] / n_vals
            
            _, l_counts = np.unique(left_side[:, 1], return_counts=True)
            _, r_counts = np.unique(right_side[:, 1], return_counts=True)
            
            l_gini = self._gini_index(l_counts)
            r_gini = self._gini_index(r_counts)
            curr_gini = left_weight * l_gini + right_weight * r_gini
            
            if curr_gini < min_gini:
                min_gini = curr_gini
                best_split_value = val
        
        return min_gini, best_split_value

    def _gini_index(self, class_counts):
        n_total = sum(class_counts)
        gi = 1
        for count in class_counts:
            gi -= (count / n_total) ** 2
        return gi

    def _feature_to_split(self, data: np.ndarray, features: list):
        _, counts = np.unique(data[:, -1], return_counts=True)
        best_gini = self._gini_index(counts)
        best_feature = None
        best_threshold = None

        for i in features:
            if np.issubdtype(data[:, i].dtype, np.number):
                gini_val, threshold = self._impurity_measure_numeric(data[:, [i, -1]])
            else:
                gini_val, threshold = self._impurity_measure_categorical(data[:, [i, -1]])
            if gini_val is None:
                continue
            if gini_val < best_gini:
                best_gini = gini_val
                best_feature = i
                best_threshold = threshold

        return best_feature, best_threshold


#this is a bit messy, will consider making a vizualization class in the future or a jupyter notebook where I run all code, but for now this will do for testing
def class_split(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = data[:, :-1]  # all columns except last
    labels = data[:, -1]     # last column is label
    return features, labels

df_heart = pd.read_csv("../datasets/heart.csv")

df_features_heart = df_heart.columns
features_heart = df_features_heart.to_numpy()

data_heart = df_heart.to_numpy()
data_heart = data_heart[1:, :]  #remove header
data_heart = data_heart.astype(np.float64)

num_runs = 5
sklearn_accuracies = []
custom_accuracies = []

for _ in range(num_runs):

    features, labels = class_split(data_heart)
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.3, random_state=None
    )
    
    #my tree
    custom_tree = DecisionTree()
    custom_tree.fit(features_train, labels_train)
    custom_predictions = custom_tree.predict(features_test)
    custom_accuracy = accuracy_score(labels_test, custom_predictions)
    custom_accuracies.append(custom_accuracy)
    
    #scikit-learn's decision tree.
    sklearn_tree = DecisionTreeClassifier()
    sklearn_tree.fit(features_train, labels_train)
    sklearn_predictions = sklearn_tree.predict(features_test)
    sklearn_accuracy = accuracy_score(labels_test, sklearn_predictions)
    sklearn_accuracies.append(sklearn_accuracy)

def plot_results(custom_accuracies, sklearn_accuracies):
    plt.figure(figsize=(10, 8))
    x_labels = [f"Iter {i+1}" for i in range(len(custom_accuracies))]
    x = np.arange(len(x_labels))
    width = 0.35

    plt.bar(x - width/2, custom_accuracies, width, label="Custom D-Tree")
    plt.bar(x + width/2, sklearn_accuracies, width, label="Scikit D-Tree")
    
    plt.xlabel("Iteration #")
    plt.ylabel("Accuracy")
    plt.title("Custom vs. Scikit-learn")
    plt.xticks(x, x_labels)
    plt.legend()
    plt.grid(axis='y')
    plt.show()

plot_results(custom_accuracies, sklearn_accuracies)
