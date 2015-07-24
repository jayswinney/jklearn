import math, random
import pandas as pd
from sys import exit
from sklearn import datasets, cross_validation
import numpy as np
from collections import Counter
from random import sample
from pprint import pprint


def prop(in_list):
    '''
    Calculates the proportion of each itemin a list. Returns a dictionary
    with the unique values and their proportion of the total
    '''

    n = float(len(in_list))
    out = Counter(in_list)
    for item in out:
        out[item] = out[item]/n
    return out

def entropy(dep):
    '''
    calculates entropy
    '''
    dep_classes = prop(dep)
    ent = 0
    for item in dep_classes:
        x = dep_classes[item]
        ent -= x * math.log(x,2)
    return ent


def info_gain_disc(ind, dep):
    '''
    calculates information gain
    for a discrete variable
    '''
    start_ent = entropy(dep)
    ind_classes = prop(ind)
    new_ent = 0
    for item in ind_classes:
        dep_item = dep[ind == item]
        new_ent += entropy(dep_item) * ind_classes[item]

    return start_ent - new_ent


def info_gain_cont(ind, dep):
    '''
    calculates information gain
    for a continuous variable
    '''
    start_ent = entropy(dep)
    df = pd.DataFrame([x for x in ind], columns = ['ind'])
    df['dep'] = dep
    df.sort('ind', inplace = True)
    ind, dep = df['ind'], df['dep']
    length = len(dep)
    new_ent = 100
    split = 0
    for i, item in enumerate(dep):
        if i == length-1:
            if item == dep[i-1] or ind[i] == ind[i-1]:
                break
        elif item == dep[i+1] or ind[i] == ind[i+1]:
            continue
        ent1=entropy(dep[:i])*((i+1)/float(length))
        ent2=entropy(dep[i:])*((length-i-1)/float(length))
        if ent1+ent2 < new_ent:
            new_ent=(ent1+ent2)
            if i == length-1:
                split = (ind[i] + ind[i-1])/2.0
            else:
                split = (ind[i] + ind[i+1])/2.0
    return start_ent - new_ent, split



def best_attr(features, dep):
    '''
    selects the best attribute to split the node on
    based on information gain
    '''
    IG = 0
    attr = None
    split = False
    for col in features:
        if np.dtype(features[col]) == 'object':
            new_IG = info_gain_disc(features[col], dep)
            if new_IG > IG:
                IG = new_IG
                attr = col
                split = False
        else:
            new_IG, new_split = info_gain_cont(features[col], dep)
            if new_IG > IG:
                IG = new_IG
                attr = col
                split = new_split
    return attr, split



def majority_value(dep):
    '''
    finds the most common value in a list
    '''
    maj = ''
    count = 0
    c_dict = Counter(dep)
    for item in c_dict:
        if c_dict[item] > count:
            count = c_dict[item]
            maj = item

    return maj

class decision_tree:
    '''
    decision tree class:
    produces a tree with a dictionary structure that follows splitting
    rules until a leaf (predicted value) is reached
    '''
    def __init__(self, features, target, min_nodes):
        self._tree = self._grow_tree(features, target, min_nodes)


    def _grow_tree(self, features, target, min_nodes):
        '''
        this function grows the tree by searching for attributes
        that have the highest information gain and splitting the data
        by those attributes untill the data is homogenous, too little
        to split or the attributes are not predctive
        '''
        if len(np.unique(target)) == 1:
            return {'type':'leaf',
                    'value':target[0]}
        if len(target) <= min_nodes:
            return {'type':'leaf',
                    'value':majority_value(target)}

        split_attr = best_attr(features, target)
        # set case for no best_attr
        if split_attr[0] == None:
            return {'type':'leaf',
                    'value':majority_value(target)}

        out = {'type':'split',
                'split_attr':split_attr[0],
                'split_value':split_attr[1],}

        if out['split_value'] != False:

            node_0 = np.array(features[out['split_attr']]>out['split_value'])
            node_1 = np.array(features[out['split_attr']]<=out['split_value'])
            out['z_node_0'] = self._grow_tree(features[node_0],
                                    target[node_0], min_nodes)

            out['z_node_1'] = self._grow_tree(features[node_1],
                                    target[node_1], min_nodes)

            return out

        else:
            classes = features[out['split_attr']].unique()

            for i,item in enumerate(classes):
                node = np.array(features[features[out['split_attr']] == item])
                out[item] = self._grow_tree(features[node],
                                                target[node], min_nodes)
            return out

    def classify(self, features):
        '''
        applies the logic stored in the tree to every
        row in the input data
        '''
        results = []
        for index, row in features.iterrows():
            results.append(self._classify(self._tree, row))
        return results

    def _classify(self, tree, record):
        '''
        classifies that data in a single row by applying
        the logic in the decision tree
        '''
        if tree['type'] == 'leaf':
            return tree['value']
        elif tree['split_value'] != False:
            if record[tree['split_attr']] > tree['split_value']:
                return self._classify(tree['z_node_0'], record)
            else:
                return self._classify(tree['z_node_1'], record)
        else:
            return self._classify(tree[record[tree['split_attr']]], record)


def bootstrap(x,target):
    '''
    returns a sample (with replacement) of the original
    dataset that has fewer columns but equal numer of rows,
    including a target array
    '''
    n = len(target)
    data = []
    column_set = sample(x.columns, len(x.columns)/2)
    x = x[column_set]
    column_set.append('target')
    x['target'] = target
    x = map(list, x.values)
    for i in range(0,n):
        data.append(random.sample(x,1)[0])

    df=pd.DataFrame(data, columns = column_set)
    target = np.array(df['target'])
    return df.drop('target', axis=1), target



class random_forest:
    '''
    random forest class:
    Used to build or make predictions from a random forest.
    Random forest are a collection of decision trees whose output is
    averaged to achieve more robust prediction than any single tree
    could provide independently
    '''
    def __init__(self, ntrees=10, min_nodes=5):
        self.ntrees = ntrees
        self.min_nodes = min_nodes

    def fit(self, features, target):
        '''
        builds the random forest
        '''
        self.forest = {}

        for i in range(0, self.ntrees):
            df, tree_target = bootstrap(features, target)
            self.forest['tree'+str(i)] = decision_tree(df, tree_target,
                                                        self.min_nodes)

    def predict(self, features):
        '''
        makes predictions using the random forest object for
        a dataset that matches the format of the training set
        '''
        results = []
        for index, row in features.iterrows():
            value = []
            for tree in self.forest:
                value.append(self.forest[tree]._classify(
                    self.forest[tree]._tree, row))
            results.append(majority_value(value))
        return results




iris = datasets.load_iris()

iris_df = pd.DataFrame(iris['data'],
columns=[x.replace(' (cm)', '') for x in iris['feature_names']])
iris_df['target'] = iris['target']

df = iris_df[iris_df['target'] < 2]
df = df.drop(['target'], axis =1)

target = iris['target']
target = target[target < 2]


x_train, x_test, y_train, y_test = cross_validation.train_test_split(
    iris_df.drop('target', axis=1), np.array(iris_df['target']))

x_train = pd.DataFrame(x_train,
columns=[x.replace(' (cm)', '') for x in iris['feature_names']])


x_test = pd.DataFrame(x_test,
columns=[x.replace(' (cm)', '') for x in iris['feature_names']])



rf_clf = random_forest(ntrees=20)

rf_clf.fit(x_train, y_train)

rf_predictions = rf_clf.predict(x_test)

print np.mean(rf_predictions == y_test)
