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