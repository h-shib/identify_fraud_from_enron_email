import matplotlib.pyplot as plt

def print_result(clf, features_train, labels_train, features_test, labels_test):
    """
    Fit classifier and print the accuracy for training/test datasets.
    """
    print "*****"
    print "Classifier:"
    print clf
    print "*****"
    clf.fit(features_train, labels_train)
    print "accuracy for training data:", clf.score(features_train, labels_train)
    print "accuracy for testing data:", clf.score(features_test, labels_test)
    print ""



def show_plot(num_of_features, precision, recall, algo_name):
    plt.plot(num_of_features, precision, marker='o', label='Precision')
    plt.plot(num_of_features, recall, marker='o', label='Recall')
    plt.title('Precision and Recall vs. Number of Features for ' + algo_name)
    plt.xlabel('K Best Features')
    plt.ylabel('Score')
    plt.legend()
    plt.show()