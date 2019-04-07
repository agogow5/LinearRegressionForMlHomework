import numpy as np
import math


def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s

class LogisticRegression():
    def __init__(self, learning_rate=.1, n_iterations=5000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def initialize_weights(self, n_features):
        # 初始化参数
        # 参数范围[-1/sqrt(N), 1/sqrt(N)]
        limit = np.sqrt(1 / n_features)
        w = np.random.uniform(-limit, limit, (n_features, 1))
        b = 0
        self.w = np.insert(w, 0, b, axis=0)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def fit(self, X, y):
        m_samples, n_features = X.shape
        self.initialize_weights(n_features)
        # 为X增加一列特征x1，x1 = 0
        X = np.insert(X, 0, 1, axis=1)
        y = np.reshape(y, (m_samples, 1))
       
#         梯度训练n_iterations轮
        for i in range(self.n_iterations):
            h_x = X.dot(self.w)
            y_pred = sigmoid(h_x)
            w_grad = X.T.dot(y_pred - y)
            self.w = self.w - self.learning_rate * w_grad
            
#         x = X
#         # 初始化
#         m, dim = x.shape
#         theta = self.w  # 参数
#         alpha = 0.01  # 学习率
#         momentum = 0.1  # 冲量
#         threshold = 0.0001  # 停止迭代的错误阈值
#         iterations = self.n_iterations  # 迭代次数
#         error = 0  # 初始错误为0

#         b1 = 0.9  # 算法作者建议的默认值
#         b2 = 0.999  # 算法作者建议的默认值
#         e = 0.00000001  #算法作者建议的默认值
#         mt = np.zeros(dim)
#         vt = np.zeros(dim)

#         for i in range(iterations):
#             j = i % m
# #             error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T,
# #                                          (np.dot(x, theta) - y))
# #             if abs(error) <= threshold:
# #                 break

#             gradient = x[j] * (np.dot(x[j], theta) - y[j])
#             mt = b1 * mt + (1 - b1) * gradient
#             vt = b2 * vt + (1 - b2) * (gradient**2)
#             mtt = mt / (1 - (b1**(i + 1)))
#             vtt = vt / (1 - (b2**(i + 1)))
# #             vtt_sqrt = np.array([np.sqrt(vtt[0]), np.sqrt(vtt[1])])  # 因为只能对标量进行开方
#             vtt_sqrt = np.sqrt(vtt)
#             theta = theta - alpha * mtt / (vtt_sqrt + e)
            
#         self.w = theta
        
        
        
        
        
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        h_x = X.dot(self.w)
        y_pred = np.round(sigmoid(h_x))
        return y_pred.astype(int)




class RandomForest():
    """Random Forest classifier. Uses a collection of classification trees that
    trains on random subsets of the data using a random subsets of the features.
    Parameters:
    -----------
    n_estimators: int
        树的数量
        The number of classification trees that are used.
    max_features: int
        每棵树选用数据集中的最大的特征数
        The maximum number of features that the classification trees are allowed to
        use.
    min_samples_split: int
        每棵树中最小的分割数，比如 min_samples_split = 2表示树切到还剩下两个数据集时就停止
        The minimum number of samples needed to make a split when building a tree.
    min_gain: float
        每棵树切到小于min_gain后停止
        The minimum impurity required to split the tree further.
    max_depth: int
        每棵树的最大层数
        The maximum depth of a tree.
    """

    def __init__(self, n_estimators=100, min_samples_split=2, min_gain=0,
                 max_depth=float("inf"), max_features=None):

        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.max_depth = max_depth
        self.max_features = max_features

        self.trees = []
        # 建立森林(bulid forest)
        for _ in range(self.n_estimators):
            tree = ClassificationTree(min_samples_split=self.min_samples_split, min_impurity=self.min_gain,
                                      max_depth=self.max_depth)
            self.trees.append(tree)

    def fit(self, X, Y):
        # 训练，每棵树使用随机的数据集(bootstrap)和随机的特征
        # every tree use random data set(bootstrap) and random feature
        sub_sets = self.get_bootstrap_data(X, Y)
        n_features = X.shape[1]
        if self.max_features == None:
            self.max_features = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            # 生成随机的特征
            # get random feature
            sub_X, sub_Y = sub_sets[i]
            idx = np.random.choice(n_features, self.max_features, replace=True)
            sub_X = sub_X[:, idx]
            self.trees[i].fit(sub_X, sub_Y)
            self.trees[i].feature_indices = idx
            print("tree", i, "fit complete")

    def predict(self, X):
        y_preds = []
        for i in range(self.n_estimators):
            idx = self.trees[i].feature_indices
            sub_X = X[:, idx]
            y_pre = self.trees[i].predict(sub_X)
            y_preds.append(y_pre)
        y_preds = np.array(y_preds).T
        y_pred = []
        for y_p in y_preds:
            # np.bincount()可以统计每个索引出现的次数
            # np.argmax()可以返回数组中最大值的索引
            # cheak np.bincount() and np.argmax() in numpy Docs
            y_pred.append(np.bincount(y_p.astype('int')).argmax())
        return y_pred

    def get_bootstrap_data(self, X, Y):

        # 通过bootstrap的方式获得n_estimators组数据
        # get int(n_estimators) datas by bootstrap

        m = X.shape[0]
        Y = Y.reshape(m, 1)

        # 合并X和Y，方便bootstrap (conbine X and Y)
        X_Y = np.hstack((X, Y))
        np.random.shuffle(X_Y)

        data_sets = []
        for _ in range(self.n_estimators):
            idm = np.random.choice(m, m, replace=True)
            bootstrap_X_Y = X_Y[idm, :]
            bootstrap_X = bootstrap_X_Y[:, :-1]
            bootstrap_Y = bootstrap_X_Y[:, -1:]
            data_sets.append([bootstrap_X, bootstrap_Y])
        return data_sets

        
#from utils import divide_on_feature, train_test_split, standardize, mean_squared_error
#from utils import calculate_entropy, accuracy_score, calculate_variance


class DecisionNode():
    """Class that represents a decision node or leaf in the decision tree
    Parameters:
    -----------
    feature_i: int
        Feature index which we want to use as the threshold measure.
    threshold: float
        The value that we will compare feature values at feature_i against to
        determine the prediction.
    value: float
        The class prediction if classification tree, or float value if regression tree.
    true_branch: DecisionNode
        Next decision node for samples where features value met the threshold.
    false_branch: DecisionNode
        Next decision node for samples where features value did not meet the threshold.
    """

    def __init__(self, feature_i=None, threshold=None,
                 value=None, true_branch=None, false_branch=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.true_branch = true_branch  # 'Left' subtree
        self.false_branch = false_branch  # 'Right' subtree


# Super class of RegressionTree and ClassificationTree
class DecisionTree(object):
    """Super class of RegressionTree and ClassificationTree.
    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    loss: function
        Loss function that is used for Gradient Boosting models to calculate impurity.
    """

    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 max_depth=float("inf"), loss=None):
        self.root = None  # Root node in dec. tree
        # Minimum n of samples to justify split
        self.min_samples_split = min_samples_split
        # The minimum impurity to justify split
        self.min_impurity = min_impurity
        # The maximum depth to grow the tree to
        self.max_depth = max_depth
        # Function to calculate impurity (classif.=>info gain, regr=>variance reduct.)
        # 切割树的方法，gini，方差等
        self._impurity_calculation = None
        # Function to determine prediction of y at leaf
        # 树节点取值的方法，分类树：选取出现最多次数的值，回归树：取所有值的平均值
        self._leaf_value_calculation = None
        # If y is one-hot encoded (multi-dim) or not (one-dim)
        self.one_dim = None
        # If Gradient Boost
        self.loss = loss

    def fit(self, X, y, loss=None):
        """ Build decision tree """
        self.one_dim = len(np.shape(y)) == 1
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data"""
        largest_impurity = 0
        best_criteria = None  # Feature index and threshold
        best_sets = None  # Subsets of the data

        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                # Iterate through all unique values of feature column i and
                # calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)

                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature
                        # index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],  # X of left subtree
                                "lefty": Xy1[:, n_features:],  # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]  # y of right subtree
                            }

        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], true_branch=true_branch, false_branch=false_branch)

        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)

    def predict_value(self, x, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at """

        if tree is None:
            tree = self.root

        # If we have a value (i.e we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value

        # Choose the feature that we will test
        feature_value = x[tree.feature_i]

        # Determine if we will follow left or right branch
        branch = tree.false_branch
        if isinstance(feature_value, int) or isinstance(feature_value, float):
            if feature_value >= tree.threshold:
                branch = tree.true_branch
        elif feature_value == tree.threshold:
            branch = tree.true_branch

        # Test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels """
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred

    def print_tree(self, tree=None, indent=" "):
        """ Recursively print the decision tree """
        if not tree:
            tree = self.root

        # If we're at leaf => print the label
        if tree.value is not None:
            print(tree.value)
        # Go deeper down the tree
        else:
            # Print test
            print("%s:%s? " % (tree.feature_i, tree.threshold))
            # Print the true scenario
            print("%sT->" % (indent), end="")
            self.print_tree(tree.true_branch, indent + indent)
            # Print the false scenario
            print("%sF->" % (indent), end="")
            self.print_tree(tree.false_branch, indent + indent)




