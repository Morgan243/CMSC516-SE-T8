import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import f1_score, make_scorer
import numpy as np
import os
import argparse

from SemEvalEight.data_prep.loaders import load_subtask1_data, load_subtask1_brown_auto_labeled
from SemEvalEight.config import tokenized_dir, ext_data_dir
from SemEvalEight import utils


def grid_search(model, param_grid,
                X, Y, scorer=f1_score,
                n_jobs=3, verbose=1):
    cv_m = GridSearchCV(model, n_jobs=n_jobs, verbose=verbose,
                        scoring=make_scorer(scorer),
                        param_grid=param_grid)

    fit_m = cv_m.fit(X, Y)
    return fit_m

def load_train_and_test_bow(train_ixes, test_ixes,
                            top_n_to_inc=None,
                            include_auto_labeled=None):
    # total of 39
    X, Y = load_subtask1_data(train_ixes,
                              tokenized_folder=tokenized_dir)

    #X_auto = open(os.path.join(ext_data_dir,
    #                           'top_3_auto_labeled_from_brown_external.txt')).readlines()
    if top_n_to_inc is not None:
        X_auto, Y_auto = load_subtask1_brown_auto_labeled(top_n=top_n_to_inc)
        #p = os.path.join(ext_data_dir, include_auto_labeled)
        #X_auto = open(p, encoding='utf-8').readlines()
        #Y_auto = np.ones(len(X_auto))

        X = np.concatenate([X, X_auto])
        Y = np.concatenate([Y, Y_auto])

    X_test, Y_test = load_subtask1_data(test_ixes,
                              tokenized_folder=tokenized_dir)


    cvec = CountVectorizer(ngram_range=(1, 2),
                           stop_words='english',
                           min_df=0)

    cvec_X = cvec.fit_transform(X).toarray()

    cvec_X_test = cvec.transform(X_test).toarray()

    return cvec_X, Y, cvec_X_test, Y_test

def run_gridsearch(model_type='dt', n_jobs=2, top_n_to_inc=0):
    # Recurse into this if given a list of model types
    if isinstance(model_type, list):
        return {mt: run_gridsearch(mt, n_jobs=n_jobs) for mt in model_type}


    file_ixs = list(range(39))
    top_n = top_n_to_inc if top_n_to_inc != 0 else None
    #auto_label_p =  'top_10_auto_labeled_from_brown_external_att.txt'
    cvec_X, Y, cvec_X_test, Y_test = load_train_and_test_bow(train_ixes=file_ixs[:23],
                                                             test_ixes=file_ixs[23:31],
                                                             top_n_to_inc=top_n)

    cv_kwargs = dict(n_jobs=n_jobs,
                     X=cvec_X, Y=Y)

    #Best Params: {'criterion': 'entropy', 'max_depth': None,
    #  'max_leaf_nodes': 12, 'min_samples_leaf': 2, 'min_samples_split': 2}
    #---
    #Best Params: {'criterion': 'entropy', 'max_depth': None,
    #  'max_leaf_nodes': 12, 'min_samples_leaf': 2, 'min_samples_split': 2}
    #---
    #Best Params: {'criterion': 'gini', 'max_depth': 25,
    # 'max_leaf_nodes': None, 'min_samples_leaf': 3, 'min_samples_split': 4}
    if model_type == 'dt':
        cv_m = grid_search(DecisionTreeClassifier(),
                           param_grid=dict(
                               criterion=['gini', 'entropy'],
                               max_depth=[None] + list(range(23, 35, 3)),
                               max_leaf_nodes=[None] + list(range(13, 22, 2)),
                               min_samples_leaf=list(range(2, 7, 2)),
                               min_samples_split=list(range(2, 7, 2))),

                           **cv_kwargs)

    elif model_type == 'gb':
        cv_m = grid_search(GradientBoostingClassifier(),
                           param_grid=dict(
                               max_depth=list(range(3, 7, 1)),
                               max_leaf_nodes=[None], #+ list(range(9, 22, 2)),
                               min_samples_leaf=list(range(2, 4, 1)),
                               min_samples_split=list(range(2, 5, 2)),

                               learning_rate=np.arange(0.7, 1.5, 0.33),
                                           n_estimators=range(100, 251, 50)),
                           **cv_kwargs)

    elif model_type == 'rf':
        cv_m = grid_search(RandomForestClassifier(),
                           param_grid=dict(max_depth=range(2, 35, 4),
                                           min_samples_split=range(2, 45, 4),
                                           n_estimators=range(25, 226, 25)),
                           **cv_kwargs)

    elif model_type == 'nb':
        cv_m = grid_search(MultinomialNB(),
                           param_grid=dict(alpha=[10**a for a in range(-3, 4, 1)]),
                           #[.5, 1.5, 3.5, 9, 17, 25]),#np.arange(.1, 3.5, .35)),
                                            **cv_kwargs)
    else:
        raise ValueError("No model type %s" % model_type)

    pred_Y = cv_m.predict(cvec_X)
    metrics = utils.binary_classification_metrics(Y, pred_Y)
    print("Best Params: %s" % str(cv_m.best_params_))
    print("Train + CV Overall Performance:")
    print("%s: %s" % (model_type, str(metrics)))

    pred_Y_test = cv_m.predict(cvec_X_test)
    test_metrics = utils.binary_classification_metrics(Y_test, pred_Y_test)
    print("Hold out performance:")
    print("%s: %s" % (model_type, str(test_metrics)))

    test_metrics = {"test_%s" % k: v for k, v in test_metrics.items()}
    metrics.update(test_metrics)
    metrics['best_params'] = dict(cv_m.best_params_)
    return metrics


def compare_auto_label():
    file_ixs = list(range(39))
    cvec_X, Y, cvec_X_test, Y_test = load_train_and_test_bow(train_ixes=file_ixs[:23],
                                                             test_ixes=file_ixs[23:31])
    preds = GradientBoostingClassifier().fit(cvec_X, Y).predict(cvec_X_test)
    #preds = DecisionTreeClassifier().fit(cvec_X, Y).predict(cvec_X_test)
    print("Labeled Only")
    metrics = utils.binary_classification_metrics(Y_test, preds)
    print(metrics)


    auto_label_p =  'top_10_auto_labeled_from_brown_external_att.txt'
    cvec_X, Y, cvec_X_test, Y_test = load_train_and_test_bow(train_ixes=file_ixs[:23],
                                                             test_ixes=file_ixs[23:31],
                                                             include_auto_labeled=auto_label_p)
    preds = GradientBoostingClassifier().fit(cvec_X, Y).predict(cvec_X_test)
    #preds = DecisionTreeClassifier().fit(cvec_X, Y).predict(cvec_X_test)
    print("AUTO LABELED")
    metrics = utils.binary_classification_metrics(Y_test, preds)
    print(metrics)



def evaluate_models_on_holdout(models_to_test, top_n_to_inc=None):
    ixes=list(range(39))
    train_X, train_Y, test_X, test_Y = load_train_and_test_bow(train_ixes=ixes[:31],
                                                               test_ixes=ixes[31:],
                                                               top_n_to_inc=top_n_to_inc)
    metrics = dict()
    for m_name, m in models_to_test:
        print("Running %s" % str(m_name))
        fit_m = m.fit(train_X, train_Y)
        preds = fit_m.predict(test_X)
        metrics[m_name] = utils.binary_classification_metrics(test_Y, preds.round())
    return metrics

def load_best_bow_ML(model_selection):
    models = list()

    if 'dt' in model_selection:
        dt_args = {'criterion': 'gini',
         'max_depth': 25,
         'max_leaf_nodes': None,
         'min_samples_leaf': 3,
         'min_samples_split': 4}
        models.append(('dt', DecisionTreeClassifier(**dt_args)))

    if 'nb' in model_selection:
        nb_args = {'alpha': 3.5}
        models.append(('nb', MultinomialNB(**nb_args)))

    if 'rf' in model_selection:
        rf_args = {'max_depth': 34,
                   'min_samples_split': 38,
                   'n_estimators': 25}
        models.append(('rf', RandomForestClassifier(**rf_args)))

    if 'gb' in model_selection:
        gb_args = dict(n_estimators=170, max_depth=5,
                    learning_rate=0.5,
                    min_samples_leaf=3, min_samples_split=4)
        models.append(('gb', GradientBoostingClassifier(**gb_args)))

    return models

if __name__ == """__main__""":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path",
                        help="Output file for serialized results",
                        default='metrics.pkl',
                        type=str)
    parser.add_argument('--n-jobs', default=2,
                        type=int)
    parser.add_argument('--top-n', default=0,
                        type=int)
    parser.add_argument('--model-type', type=str,
                        default='nb',
                        help='One of, or multiple comma separated, [nb, dt, rf, gb] ')
    parser.add_argument('--eval-best', action='store_true',
                        default=False)

    args = parser.parse_args()
    models = args.model_type.split(',')

    if args.eval_best:
        print(models)
        model_objs = load_best_bow_ML(models)
        print(model_objs)
        eval_res = evaluate_models_on_holdout(model_objs, top_n_to_inc=args.top_n if args.top_n != 0 else None)
        print(eval_res)
    else:

        print("Running %s" % models)
        metrics = run_gridsearch(model_type=models, n_jobs=args.n_jobs)
        print("")
        print(metrics)

        with open(args.output_path, 'wb') as f:
            pickle.dump(metrics, f)
