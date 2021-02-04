#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np

from io import StringIO
from functools import partial


# 1. Use the Tree data structure below; write code to build the tree from figure 1.2 in Daumé.

# In[12]:


class Tree:
  '''Create a binary tree; keyword-only arguments `data`, `left`, `right`.

  Examples:
    l1 = Tree.leaf("leaf1")
    l2 = Tree.leaf("leaf2")
    tree = Tree(data="root", left=l1, right=Tree(right=l2))
  '''
  @staticmethod
  def leaf(data):
    '''Create a leaf tree
    '''
    return Tree(data=data)

  # pretty-print trees
  def __repr__(self):
    if self.is_leaf():
      return "Leaf(%r)" % self.data
    else:
      return "Tree(%r) { left = %r, right = %r }" % (self.data, self.left, self.right) 

  # all arguments after `*` are *keyword-only*!
  def __init__(self, *, data = None, left = None, right = None):
    self.data = data
    self.left = left
    self.right = right

  def is_leaf(self):
    '''Check if this tree is a leaf tree
    '''
    return self.left == None and self.right == None

  def children(self):
    '''List of child subtrees
    '''
    return [x for x in [self.left, self.right] if x]

  def depth(self):
    '''Compute the depth of a tree
    A leaf is depth-1, and a child is one deeper than the parent.
    '''
    return max([x.depth() for x in self.children()], default=0) + 1


# In[13]:


## create leaves:
l1, l2, l3, l4, l5 = (Tree.leaf("like"), Tree.leaf("like"),
Tree.leaf('nah'), Tree.leaf("nah"), Tree.leaf("like"))


# In[14]:


node1 = Tree(data="morning?", left=l2, right=l3)
node2 = Tree(data="likedOtherSys?", left=l4, right=l5)
node3 = Tree(data="takenOtherSys?", left=node1, right=node2)
root = Tree(data="isSystems?", left=l1, right=node3)


# 2. In your python code, load the following dataset and add a boolean "ok" column, where "True" means the rating is non-negative and "False" means the rating is negative.

# In[15]:


csv_string = """rating,easy,ai,systems,theory,morning
 2,True,True,False,True,False
 2,True,True,False,True,False
 2,False,True,False,False,False
 2,False,False,False,True,False
 2,False,True,True,False,True
 1,True,True,False,False,False
 1,True,True,False,True,False
 1,False,True,False,True,False
 0,False,False,False,False,True
 0,True,False,False,True,True
 0,False,True,False,True,False
 0,True,True,True,True,True
-1,True,True,True,False,True
-1,False,False,True,True,False
-1,False,False,True,False,True
-1,True,False,True,False,True
-2,False,False,True,True,False
-2,False,True,True,False,True
-2,True,False,True,False,False
-2,True,False,True,False,True"""


# In[16]:


df = pd.read_csv(StringIO(csv_string))


# In[17]:


df["ok"] = df["rating"] >= 0


# 3. Write a function which takes a feature and computes the performance of the corresponding single-feature classifier:

# In[18]:


def single_feature_score(data, goal, feature):
  pos_class = goal[data[feature] == True].value_counts().argmax()
  neg_class = goal[data[feature] == False].value_counts().argmax()
  clf = lambda x: pos_class if x else neg_class

  predicted = data[feature].apply(clf)
  acc = (predicted == goal).mean()

  return acc


# Use this to find the best feature:

# In[19]:


def best_feature(data, goal, features):
  # optional: avoid the lambda using `functools.partial`
  return max(features, key=lambda f: single_feature_score(data, goal, f))


# In[20]:


X = df.drop(["ok","rating"], axis=1)
y = df["ok"]


# In[22]:


print(f"The best feature for single-feature classifier is '{best_feature(X, y, X.columns)}''")


# Which feature is best? Which feature is worst?

# We can see that the best feature is "systems". Now. let's look at other features:

# In[23]:


def feature_scores(data, goal, features):
    return sorted([(feat, single_feature_score(data, goal, feat)) for feat in features],
    key=lambda x: x[1])


# In[24]:


score_dict = feature_scores(X, y, X.columns)
print(f"Scores for the features in the decision tree: {score_dict}")


# In[37]:


best_performance  = max(score_dict, key = lambda x: x[1])[1]


# The worst feature is "easy"

# 4. Implement the DecisionTreeTrain and DecisionTreeTest algorithms from Daumé, returning Trees. (Note: our dataset and his are different; we won't get the same tree.)
# 
# How does the performance compare to the single-feature classifiers?

# In[26]:


def nanz(x):
    if np.isnan(x):
        return 0
    return x

def decision_tree_train(X, y, features):
    guess = y.value_counts().argmax()
    if len(y.unique()) == 1 or not features:
        return Tree.leaf(guess)
    else:
        score = dict()
        for feat in features:
            no = X[feat] == False
            yes = X[feat] == True
            score[feat] = nanz(y[no].value_counts().max())+nanz(y[yes].value_counts().max())
        feat = pd.Series(score).argmax()
        no = X[feat] == False
        yes = X[feat] == True
        X_no, y_no = X[no], y[no]
        X_yes, y_yes = X[yes], y[yes]
        if len(X_no) == 0 or len(X_yes) == 0:
            return Tree.leaf(guess)
        left = decision_tree_train(X_no, y_no, [f for f in features if f!=feat])
        right = decision_tree_train(X_yes, y_yes, [f for f in features if f!=feat])
        return Tree(data=feat, left=left, right=right)


# In[27]:


t = decision_tree_train(X, y, list(X.columns))


# In[28]:


def decision_tree_test(tree, test_point):
    if tree.right is None and tree.left is None:
        return tree.data
    else:
        if not test_point[tree.data]:
            return decision_tree_test(tree.left, test_point)
        else:
            return decision_tree_test(tree.right, test_point)


# In[29]:


predict = lambda x: decision_tree_test(t, x)
predicted = X.apply(predict, axis=1)


# In[30]:


improved_performance = (predicted == y).mean()


# In[40]:


print(f"Now the performance is {improved_performance} which is by {improved_performance-best_performance} higher than the best single-feature performance.")


# Now the score is higher by 5%

# 5. Add an optional maxdepth parameter to DecisionTreeTrain, which limits the depth of the tree produced. Plot performance against maxdepth.

# In[86]:


def nanz(x):
    if np.isnan(x):
        return 0
    return x

def decision_tree_train(X, y, features, max_depth, depth=0):
    guess = y.value_counts().argmax()
    if len(y.unique()) == 1 or not features or depth==max_depth:
        return Tree.leaf(guess)
    else:
        score = dict()
        for feat in features:
            no = X[feat] == False
            yes = X[feat] == True
            score[feat] = nanz(y[no].value_counts().max())+nanz(y[yes].value_counts().max())
        feat = pd.Series(score).argmax()
        no = X[feat] == False
        yes = X[feat] == True
        X_no, y_no = X[no], y[no]
        X_yes, y_yes = X[yes], y[yes]
        if len(X_no) == 0 or len(X_yes) == 0:
            return Tree.leaf(guess)
        left = decision_tree_train(X_no, y_no, [f for f in features if f!=feat], max_depth, depth+1)
        right = decision_tree_train(X_yes, y_yes, [f for f in features if f!=feat], max_depth, depth+1)
        return Tree(data=feat, left=left, right=right)


# In[91]:


def predict(tree, X):
    predictor = lambda x: decision_tree_test(tree, x)
    return X.apply(predictor, axis=1)

def score(tree, X, y):
    predicted = predict(tree, X)
    return (predicted == y).mean()

features = list(X.columns)
depth = [i for i in range(len(features)+1)]
score = [score(decision_tree_train(X, y, features, i), X, y) for i in depth]


# In[97]:


import matplotlib.pyplot as plt
plt.plot(depth, score)
plt.xlabel("depth")
plt.ylabel("accuracy")
plt.show()


# In[ ]:




