from collections import Counter
import numpy as np


def mse(y):

    try:
        y_mean = y.mean()
    except AttributeError:
        y = np.array(y)
        y_mean = y.mean()

    return ((y - y_mean) ** 2).mean()


class DecisionRule:
    def __init__(self, k, bound):
        self.k = k
        self.bound = bound

    def __call__(self, x):
        return not x[self.k] < self.bound


class Node:
    EPS = 1e-7

    def __init__(self, tree, depth=0):
        self.depth = depth
        self.tree = tree
        self.edges = list()
        self.rule = None
        self.indices = list()

    def get_class(self, x):
        if not self.edges:
            return self.tree.y[self.indices].mean()

        return self.edges[self.rule(x)].get_class(x)

    @staticmethod
    def _criterion(left, right, root):
        l = len(left)
        r = len(right)
        m = len(root)

        return l / m * mse(left) + r / m * mse(right)

    def _get_classes(self, indices):
        return Counter(self.tree.y[indices])

    def _split_indices(self, indices, bound, k):
        left = list(filter(
            lambda idx: self.tree.X[idx][k] < bound - Node.EPS,
            indices))
        right = list(filter(
            lambda idx: not self.tree.X[idx][k] < bound - Node.EPS,
            indices))

        return left, right

    def learn(self, indices):
        self.indices = indices

        if self.depth == self.tree.max_depth \
                or len(indices) < self.tree.min_items:
            return

        best_k = -1
        best_bound = -1
        c_value = float('Inf')

        for k in range(len(self.tree.bounds)):
            for bound in self.tree.bounds[k]:
                left, right = self._split_indices(indices, bound, k)

                new_c_value = self._criterion(left, right, indices)
                if new_c_value < c_value:
                    best_k, best_bound = k, bound
                    c_value = new_c_value

        if best_k == -1:
            return

        self.rule = DecisionRule(best_k, best_bound)

        left, right = self._split_indices(indices, best_bound, best_k)

        l_node = Node(self.tree, self.depth + 1)
        r_node = Node(self.tree, self.depth + 1)

        l_node.learn(left)
        r_node.learn(right)
        self.edges = [l_node, r_node]


class DecisionTree:
    def __init__(self, max_depth=3, min_items=5):
        self.root = Node(self)
        self.X = None
        self.y = None
        self.max_depth = max_depth
        self.min_items = min_items

        self.bounds = None

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.bounds = list()

        for k in range(X.shape[1]):
            self.bounds.append(np.linspace(np.min(self.X[:, k]),
                                           np.max(self.X[:, k]), num=10))

        self.root.learn(list(range(len(X))))

    def predict(self, X):
        return [self.root.get_class(x) for x in np.asarray(X)]
