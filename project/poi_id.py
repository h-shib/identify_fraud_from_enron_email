#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier
from sklearn.grid_search import GridSearchCV
from time import time

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'exercised_stock_options', 'deferral_payments']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
# remove outlier
my_dataset.pop('TOTAL', 0)

# add new feature, fraction of mails send to poi.
for key, value in my_dataset.iteritems():
    if value['to_messages'] != 'NaN' and value['from_this_person_to_poi'] != 'NaN':
        my_dataset[key]['fraction_to_poi'] = float(value['from_this_person_to_poi']) / value['to_messages']
    else:
        my_dataset[key]['fraction_to_poi'] = 0
features_list.append('fraction_to_poi')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

"""
Tuned parameter with GridSearchCV below.

param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2, 5, 10, 20],
              "max_depth": [None, 2, 5, 10],
              "min_samples_leaf": [1, 5, 10],
              "max_leaf_nodes": [None, 5, 10, 20],
              }

clf = tree.DecisionTreeClassifier()
clf = GridSearchCV(clf, param_grid)
clf.fit(features_train, labels_train)
print clf.score(features_test, labels_test)
"""

from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None, max_features=None,
                             max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=5)

clf.fit(features_train, labels_train)
print clf.feature_importances_

print "accuracy for training data:", clf.score(features_train, labels_train)
print "accuracy for testing data:", clf.score(features_test, labels_test)

t0 = time()
test_classifier(clf, my_dataset, features_list, folds = 1000)
print "time:", time() - t0
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)