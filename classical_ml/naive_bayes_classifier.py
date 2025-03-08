from math import e, pi, sqrt

import numpy as np
import pandas as pd


class NaiveBayesClassifier:
    """Implements Naïve Bayes classification with Gaussian probability density."""

    def __init__(self):
        self.class_statistics = {}

    def __calculate_feature_statistics(self, feature_data: np.ndarray, total_samples: int):
        """Calculates mean, std deviation, and prior probability for a given class."""
        mean = np.mean(feature_data, axis=0, dtype=np.float64)
        std_dev = np.std(feature_data, axis=0, dtype=np.float64)
        prior = feature_data.shape[0] / total_samples
        return mean, std_dev, prior

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray):
        """Trains the Naïve Bayes classifier by computing feature statistics per class."""
        unique_classes = np.unique(train_labels)
        for class_label in unique_classes:
            class_features = train_features[train_labels == class_label]
            self.class_statistics[class_label] = self.__calculate_feature_statistics(class_features, len(train_features))

    def __gaussian_probability_density_function(self, x: np.ndarray, means: np.ndarray, stdevs: np.ndarray):
        """Computes Gaussian probability density function for input features."""
        prob_density = np.zeros(len(means), dtype=np.float64)
        for i in range(len(means)):
            prob_density[i] = (e ** ((-(x[i] - means[i]) ** 2) / (2 * (stdevs[i] ** 2)))) / sqrt(2 * pi * (stdevs[i] ** 2)) #not the prettiest but hey, it works
        return prob_density

    def __predict_probabilities(self, X: np.ndarray):
        """Returns probability estimates for each class given the input features."""
        prediction_probabilities = []
        
        for data_point in X:
            class_probs = []
            for class_label, (means, stddevs, prior) in self.class_statistics.items():
                gaussian_probability = self.__gaussian_probability_density_function(data_point, means, stddevs)
                class_probs.append(np.prod(gaussian_probability) * prior)

            normalized_predictions = [p / np.sum(class_probs) for p in class_probs]
            prediction_probabilities.append(normalized_predictions)

        return np.array(prediction_probabilities)

    def predict(self, X: np.ndarray):
        """Predicts the class label for each sample based on maximum probability."""
        probabilities = self.__predict_probabilities(X)
        return np.argmax(probabilities, axis=1)

def train_test_split(class_data: np.ndarray, test_size=0.2) -> tuple[np.ndarray, np.ndarray]:
    """Splits the dataset into training and testing sets."""
    mask = np.ones(class_data.shape[0], dtype=bool)
    test_indices = np.random.choice(class_data.shape[0], size=int(class_data.shape[0] * test_size), replace=False)
    mask[test_indices] = False
    return class_data[mask], class_data[~mask]

def class_split(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Splits dataset into features and labels."""
    return data[:, :-1], data[:, -1].astype(np.int32)

data = pd.read_csv("../datasets/iris.csv").to_numpy()

# Convert categorical labels to numeric values
mapped, index, unique_arr = np.unique(data[:, -1], return_index=True, return_inverse=True)
data[:, -1] = unique_arr

# Train-test split
train_data, test_data = train_test_split(data)

# Extract features and labels
train_features, train_labels = class_split(train_data)
test_features, test_labels = class_split(test_data)

# Train Naïve Bayes Classifier
nb_model = NaiveBayesClassifier()
nb_model.fit(train_features, train_labels)

# Make predictions
predictions = nb_model.predict(test_features)

# Print predictions
print("Predicted Labels:", predictions)
print("Actual Labels:", test_labels)

# Print calculated priors
#print("Class priors:", nb_model.class_counts)