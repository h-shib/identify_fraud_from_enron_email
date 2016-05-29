#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL', 0)
my_dataset = data_dict


for name, value in my_dataset.iteritems():
    data = my_dataset[name]
    matplotlib.pyplot.scatter(data['expenses'], data['long_term_incentive'])

matplotlib.pyplot.show()

for name, value in my_dataset.iteritems():
    data = my_dataset[name]
    if data['long_term_incentive'] > 1500000 and data['long_term_incentive'] != 'NaN':
        print name, value, "\n"