from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB
import numpy as np

from SemEvalEight.data_prep.loaders import load_subtask1_data
from SemEvalEight.config import tokenized_dir
from SemEvalEight import utils

def grid_search(model, param_grid,
                X, Y,
                n_jobs=3, verbose=1):
    cv_m = GridSearchCV(model, n_jobs=n_jobs, verbose=verbose,
                        param_grid=param_grid)

    fit_m = cv_m.fit(X, Y)
    return fit_m

def run(model_type='dt', n_jobs=2):
    # Recurse into this if given a list of model types
    if isinstance(model_type, list):
        return {mt: run(mt, n_jobs=n_jobs) for mt in model_type}

    # total of 39
    file_ixs = list(range(39))
    X, Y = load_subtask1_data(file_ixs[:20],
                              tokenized_folder=tokenized_dir)
    X_test, Y_test = load_subtask1_data(file_ixs[20:],
                              tokenized_folder=tokenized_dir)

    cvec = CountVectorizer(ngram_range=(1, 2))

    cvec_X = cvec.fit_transform(X)
    cvec_X = cvec_X.toarray()

    cvec_X_test = cvec.transform(X_test)
    cvec_X_test = cvec_X_test.toarray()

    cv_kwargs = dict(n_jobs=n_jobs,
                     X=cvec_X, Y=Y)


    if model_type == 'dt':
        cv_m = grid_search(DecisionTreeClassifier(),
                           param_grid=dict(max_depth=range(2, 20, 2),
                                           min_samples_split=range(1, 20, 2)),
                           **cv_kwargs)

    elif model_type == 'gb':
        cv_m = grid_search(GradientBoostingClassifier(),
                           param_grid=dict(learning_rate=np.arange(0.1, 1.5, 0.2),
                                           n_estimators=range(25, 126, 25)),
                           **cv_kwargs)
    elif model_type == 'rf':
        cv_m = grid_search(RandomForestClassifier(),
                           param_grid=dict(max_depth=range(2, 25, 3),
                                           min_samples_split=range(2, 25, 3),
                                           n_estimators=range(25, 126, 25)),
                           **cv_kwargs)
    elif model_type == 'nb':
        cv_m = grid_search(BernoulliNB(),
                           param_grid=dict(alph=np.arange(.1,2.0, .1)),
                                            **cv_kwargs)
    else:
        raise ValueError("No model type %s" % model_type)

    pred_Y = cv_m.predict(cvec_X)
    metrics = utils.binary_classification_metrics(Y, pred_Y)
    print("Train + CV Overall Performance:")
    print("%s: %s" % (model_type, str(metrics)))

    pred_Y_test = cv_m.predict(cvec_X_test)
    metrics = utils.binary_classification_metrics(Y_test, pred_Y_test)
    print("Hold out performance:")
    print("%s: %s" % (model_type, str(metrics)))

    return metrics

if __name__ == """__main__""":
    mt = [
        'rf',
        'gb'
    ]
    metrics = run(model_type=mt, n_jobs=6)

