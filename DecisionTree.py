from typing import List
from collections import Counter
import math

class Node:
    """
    Represents a node in a decision tree.

    Attributes:
    - split_dim: index of the feature to split on
    - split_point: value to split the feature
    - label: majority class label of the data at this node
    - left: left child (for values <= split_point)
    - right: right child (for values > split_point)
    """

    def __init__(self):
        self.split_dim = -1
        self.split_point = -1
        self.label = -1
        self.left = None
        self.right = None

class Solution:
    """
    Decision tree classifier using recursive binary splitting with entropy-based information gain.
    """

    def fit(self, train_data: List[List[float]], train_label: List[int]) -> None:
        """Trains the decision tree by creating the root and starting recursive splitting."""
        self.root = Node()
        self._split_node(self.root, train_data, train_label, depth=0)

    def split_info(self, data: List[List[float]], labels: List[int], split_dim: int, split_point: float) -> float:
        """
        Computes weighted entropy after splitting on a given feature and point.
        """
        left_labels = []
        right_labels = []
        for x, y in zip(data, labels):
            if x[split_dim] <= split_point:
                left_labels.append(y)
            else:
                right_labels.append(y)

        total = len(labels)
        left_weight = len(left_labels) / total
        right_weight = len(right_labels) / total

        return left_weight * self._entropy(left_labels) + right_weight * self._entropy(right_labels)

    def classify(self, train_data: List[List[float]], train_label: List[int], test_data: List[List[float]]) -> List[int]:
        """
        Trains the model and predicts labels for test data using the trained decision tree.
        """
        self.fit(train_data, train_label)
        predictions = []
        for x in test_data:
            node = self.root
            # Traverse the tree until a leaf node is reached
            while node.left and node.right:
                if x[node.split_dim] <= node.split_point:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.label)
        return predictions

    def _entropy(self, labels: List[int]) -> float:
        """
        Calculates the entropy of a list of class labels.
        """
        total = len(labels)
        label_counts = Counter(labels)
        ent = 0.0
        for count in label_counts.values():
            p = count / total
            ent += -p * math.log2(p)
        return ent

    def _majority_label(self, labels: List[int]) -> int:
        """
        Returns the majority label (ties broken by smaller value).
        """
        counts = Counter(labels)
        max_count = max(counts.values())
        candidates = [label for label, count in counts.items() if count == max_count]
        return min(candidates)

    def _is_better_gain(self, gain, best_gain, dim, point, best_dim, best_point):
        """
        Compares two gains to determine which split is better.
        """
        if gain > best_gain:
            return True
        if gain == best_gain:
            if dim < best_dim:
                return True
            if dim == best_dim and point < best_point:
                return True
        return False

    def _get_split_points(self, values: List[float]) -> List[float]:
        """
        Computes candidate split points from sorted unique feature values.
        """
        sorted_vals = sorted(set(values))
        return [(sorted_vals[i] + sorted_vals[i + 1]) / 2 for i in range(len(sorted_vals) - 1)]

    def _split_node(self, node: Node, data: List[List[float]], labels: List[int], depth: int):
        """
        Recursively splits the data to build the tree up to max depth = 2 or pure class labels.
        """
        node.label = self._majority_label(labels)

        # Stopping conditions: max depth reached or all labels are the same
        if depth == 2 or len(set(labels)) == 1:
            return

        best_gain = -1
        best_dim = -1
        best_point = -1.0
        current_entropy = self._entropy(labels)

        # Try all dimensions and possible split points
        for dim in range(len(data[0])):
            split_points = self._get_split_points([x[dim] for x in data])
            for point in split_points:
                info = self.split_info(data, labels, dim, point)
                gain = current_entropy - info
                if self._is_better_gain(gain, best_gain, dim, point, best_dim, best_point):
                    best_gain, best_dim, best_point = gain, dim, point

        # If no gain is possible, return as a leaf
        if best_gain < 0:
            return

        node.split_dim = best_dim
        node.split_point = best_point

        # Split the dataset
        left_data, left_labels = [], []
        right_data, right_labels = [], []
        for x, y in zip(data, labels):
            if x[best_dim] <= best_point:
                left_data.append(x)
                left_labels.append(y)
            else:
                right_data.append(x)
                right_labels.append(y)

        # Recursively split children
        node.left = Node()
        node.right = Node()
        self._split_node(node.left, left_data, left_labels, depth + 1)
        self._split_node(node.right, right_data, right_labels, depth + 1)
