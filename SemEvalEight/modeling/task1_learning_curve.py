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
    -- fatima
    Generates a plot of the learning curve with the test data (F1 score and standard deviation)
    Iterates through the training samples 100 times, and each time the sample gradually increases.
    For each number of samples, make a 3-fold cross-validation. The mean is used for the graphic.


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

        # We shuffle the training data on each iteration
        X, y = shuffle(X, y, random_state=0)

        partial_results = []
        for r in range(0, 3):
            #X_tmp, y_tmp = resample(X, y, n_samples=X_length_count, random_state=0)

            # Generates a random array of index numbers to take it from the training set.
            idx = np.random.choice(np.arange(len(X)), number_samples, replace=False)
            X_tmp = X[idx,:]
            y_tmp = np.take(y, idx)

            classifier.fit(X_tmp, y_tmp)
            y_predicted = classifier.predict(X_test)

            metrics = utils.binary_classification_metrics(y_test, y_predicted)
            print(metrics)

            # We save the F1 score to calculate the mean
            partial_results.append(metrics["f1"])

        # Calculate mean and sdt from k-cross
        mean_f1 = np.mean(partial_results)
        print("The mean is -> ", mean_f1)

        sdt_f1 = np.std(partial_results, ddof=1)
        print("The sdt is -> ", sdt_f1)

        # Add results to array to generate plot at the end
        results.update({str(number_samples):dict(mean_f1=mean_f1, sdt_f1=sdt_f1)})
        number_samples = number_samples + X_length_inc

    print("Plotting results...")

    samples_range = []
    mean_f1 = []
    sdt_f1 = []

    for key, value in results.items():
        samples_range.append(key)
        mean_f1.append(value["mean_f1"])
        sdt_f1.append(value["sdt_f1"])

    # Configure plot settings
    plt.figure(figsize=(12, 9))
    ax = plt.subplot(111)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # Use matplotlib's fill_between() call to create error bars. Use the dark blue "#3F5D7D" color.
    plt.fill_between(samples_range, np.array(mean_f1) - np.array(sdt_f1),
                     np.array(mean_f1) + np.array(sdt_f1), color="#3F5D7D")

    # Plot the means as a white line
    plt.plot(samples_range, mean_f1, color="white", lw=2)

    plt.title("Learning curve - GradientBoosting", fontsize=22)
    plt.ylabel("F1 score", fontsize=16)
    plt.xlabel("Training samples", fontsize=16)

    plt.show()

    print(results)
    print("mean = ", mean_f1)
    print("sdt = ", sdt_f1)
    print("training_samples = ", samples_range)
    return results



# total of 39
# we took 70% of the files for training, and 30% for testing.
file_ixs = list(range(39))

print ("Start time: ", time.strftime("%H:%M:%S"))

X, y = load_subtask1_data(file_ixs[:28],
                          tokenized_folder=tokenized_dir)
X_test, y_test = load_subtask1_data(file_ixs[28:],
                                    tokenized_folder=tokenized_dir)

cvec = CountVectorizer(ngram_range=(1, 2))

cvec_X = cvec.fit_transform(X)
cvec_X = cvec_X.toarray()

cvec_X_test = cvec.transform(X_test)
cvec_X_test = cvec_X_test.toarray()

classifier = DecisionTreeClassifier(criterion='gini', max_depth=25, max_leaf_nodes=None, min_samples_leaf=3, min_samples_split=4)
#classifier = GradientBoostingClassifier()

calculate_learning_curve(classifier, cvec_X, y, cvec_X_test, y_test)

print ("End time", time.strftime("%H:%M:%S"))