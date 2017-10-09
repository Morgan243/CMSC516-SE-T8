import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier

from SemEvalEight.data_prep.loaders import load_subtask1_data
from SemEvalEight.config import tokenized_dir
from SemEvalEight import utils

def calculate_learning_curve(classifier, X, y, X_test, y_test):
    """

    :param classifier: Object type that implements the "fit" and "predict" methods.
    :param X: Training dataset.
    :param y: Target relative to X.
    :param X_test: Testing dataset.
    :param y_test: Target relative to X_test.
    :return: Dictionary where key = total training samples, and value = metrics from test.
    """

    results = dict()
    X_length = len(X)
    X_length_inc = X_length // 100
    number_samples = X_length_inc

    for x in range(0, 100):
        print("This is the %s iteration" % str(x + 1))
        print("Total training samples is %s" % str(number_samples))

        X, y = shuffle(X, y, random_state=0)

        partial_results = []
        for r in range(0, 3):
            #X_tmp, y_tmp = resample(X, y, n_samples=X_length_count, random_state=0)

            idx = np.random.choice(np.arange(len(X)), number_samples, replace=False)
            X_tmp = X[idx,:]
            y_tmp = np.take(y, idx)

            classifier.fit(X_tmp, y_tmp)
            y_predicted = classifier.predict(X_test)

            metrics = utils.binary_classification_metrics(y_test, y_predicted)
            print(metrics)

            partial_results.append(metrics["f1"])


        mean_f1 = np.mean(partial_results)
        print("The mean is -> %f", mean_f1)

        sdt_f1 = np.std(partial_results, ddof = 1)
        print("The sdt is -> %f", sdt_f1)

        results.update({str(number_samples):dict(mean_f1=mean_f1, sdt_f1 = sdt_f1)})
        number_samples = number_samples + X_length_inc

    print("Plotting results...")

    samples_range = []
    mean_f1 = []
    sdt_f1 = []

    for key, value in results.items():
        samples_range.append(key)
        mean_f1.append(value["mean_f1"])
        sdt_f1.append(value["sdt_f1"])

    plt.figure()
    plt.title("Learning curve - Decision tree")
    plt.xlabel("Training samples")
    plt.ylabel("F1 score")
    plt.plot(samples_range, mean_f1)
    plt.plot(samples_range, sdt_f1)

    plt.show()

    print(results)
    return results



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# #digits = load_digits()
# #X, y = digits.data, digits.target
#
# file_ixs = list(range(39))
# X, y = load_subtask1_data(file_ixs[:20],
#                           tokenized_folder=tokenized_dir)
# #X_test, Y_test = load_subtask1_data(file_ixs[20:],
# #                                    tokenized_folder=tokenized_dir)
#
# cvec = CountVectorizer(ngram_range=(1, 2))
#
# cvec_X = cvec.fit_transform(X)
# cvec_X = cvec_X.toarray()
#
# #cvec_X_test = cvec.transform(X_test)
# #cvec_X_test = cvec_X_test.toarray()
#
# #cv_kwargs = dict(n_jobs=n_jobs,
# #                 X=cvec_X, Y=Y)
#
#
# title = "Learning Curves (Naive Bayes)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
# estimator = GaussianNB()
# print("Starting plot learning curve - GaussianNB")
# plot_learning_curve(estimator, title, cvec_X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)
#
# title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# # cv = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
# # estimator = SVC(gamma=0.001)
# # print("Starting plot learning curve - SVC")
# # plot_learning_curve(estimator, title, cvec_X, y, (0.7, 1.01), cv=cv, n_jobs=1)
#
#
# title = "Learning Curves (Decision Tree)"
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = DecisionTreeClassifier()
# print("Starting plot learning curve - DecisionTree")
# plot_learning_curve(estimator, title, cvec_X, y, (0.7, 1.01), cv=cv, n_jobs=1)
#
# print("Let's show it")
# plt.show()


# total of 39
file_ixs = list(range(39))
X, y = load_subtask1_data(file_ixs[:28],
                          tokenized_folder=tokenized_dir)
X_test, y_test = load_subtask1_data(file_ixs[28:],
                                    tokenized_folder=tokenized_dir)

print ("Start time: ", time.strftime("%H:%M:%S"))

cvec = CountVectorizer(ngram_range=(1, 2))

cvec_X = cvec.fit_transform(X)
cvec_X = cvec_X.toarray()

cvec_X_test = cvec.transform(X_test)
cvec_X_test = cvec_X_test.toarray()

classifier = DecisionTreeClassifier(criterion='gini', max_depth=25, max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=4)
#classifier = GradientBoostingClassifier()

calculate_learning_curve(classifier, cvec_X, y, cvec_X_test, y_test)

print ("End time", time.strftime("%H:%M:%S"))