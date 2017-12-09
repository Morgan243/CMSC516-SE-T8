import argparse
import pandas as pd
from pprint import pprint
from SemEvalEight.modeling.task2.task2_feature_extraction import create_model_ready
from SemEvalEight.utils import binary_classification_metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, make_scorer, f1_score

def basic_grid_search(window_size=0, embed_type='glove', model='gb'):
    model_df, count_vec_model, model_features = create_model_ready(ixes=range(41), window_size=window_size,
                                                                   embed_dim=100, embed_type=embed_type,
                                                                   neg_include_proba=.25)

    cv_df, cv_vec_model, cv_features = create_model_ready(ixes=range(41, 53),
                                                          embed_dim=100, embed_type=embed_type,
                                                          window_size=window_size,
                                                          count_vec_model=count_vec_model,
                                                          neg_include_proba=1.)
    train_df = pd.concat([model_df, cv_df])

# Since these are built up seperately, possible for differing features, so do quick check
    assert len(model_features) == len(cv_features)

    gb_gs = GridSearchCV(GradientBoostingClassifier(),
                         param_grid=dict(learning_rate=[0.1], n_estimators=[300, 400, 500],
                                         min_samples_split=[2],
                                         min_samples_leaf=[1],
                                         max_depth=[4, 5],
                                         max_leaf_nodes=[None, 20]
                                         ),
                         scoring=make_scorer(f1_score), n_jobs=3, verbose=1
                         )

    #gb_gs.fit(model_df[model_features], model_df.is_target)
    gb_gs.fit(train_df[model_features], train_df.is_target)

    gs_gb_preds = gb_gs.predict_proba(cv_df[model_features])

    print(gb_gs.best_estimator_)
    print(classification_report(cv_df.is_target, gs_gb_preds[:, 1] > 0.5))

def tune_parameters(target_label, window_sizes, embed_dims, model=None):
    if model is None:
        model = GradientBoostingClassifier(n_estimators=400, verbose=1)

    results = list()
    for ws in window_sizes:
        for ed in embed_dims:
            print("Window size: %d" % ws)
            model_df, count_vec_model, model_features = create_model_ready(ixes=range(41),
                                                                           window_size=ws,
                                                                           target_type_label=target_label,
                                                                           neg_include_proba=.25, embed_dim=ed)

            cv_df, cv_vec_model, cv_features = create_model_ready(ixes=range(41, 53),
                                                                  window_size=ws,
                                                                  target_type_label=target_label,
                                                                  count_vec_model=count_vec_model,
                                                                  neg_include_proba=1., embed_dim=ed)

            fm = model.fit(model_df[model_features], model_df.is_target)
            fm_preds = fm.predict_proba(cv_df[cv_features])

            cv_metrics = binary_classification_metrics(cv_df.is_target, fm_preds[:, 1] > 0.5)
            cv_metrics['window_size'] = ws

            print(classification_report(cv_df.is_target, fm_preds[:, 1] > 0.5))
            print(cv_metrics)
            results.append(cv_metrics)

    return results

def eval_on_holdout(target_label, window_size, embed_dim, embed_type, model=None, eval_holdout=False):
    if model is None:
        model = GradientBoostingClassifier(n_estimators=400, verbose=1)

    print("Window size: %d" % window_size)
    model_df, count_vec_model, model_features = create_model_ready(ixes=range(41),
                                                                   window_size=window_size,
                                                                   target_type_label=target_label,
                                                                   neg_include_proba=.35,
                                                                   embed_dim=embed_dim, embed_type=embed_type)

    cv_df, cv_vec_model, cv_features = create_model_ready(ixes=range(41, 53),
                                                          window_size=window_size,
                                                          target_type_label=target_label,
                                                          count_vec_model=count_vec_model,
                                                          neg_include_proba=1.,
                                                          embed_dim=embed_dim, embed_type=embed_type)

    holdout_df, holdout_vec_model, holdout_features = create_model_ready(ixes=range(53, 65),
                                                                         window_size=window_size,
                                                                         target_type_label=target_label,
                                                                         count_vec_model=count_vec_model,
                                                                         neg_include_proba=1.,
                                                                         embed_dim=embed_dim, embed_type=embed_type)

    if eval_holdout:
        train_df = pd.concat([model_df, cv_df]).sample(frac=1.)
        test_df = holdout_df
    else:
        train_df = model_df
        test_df = cv_df

    fm = model.fit(train_df[model_features], train_df.is_target)
    fm_preds = fm.predict_proba(test_df[model_features])


    metrics = binary_classification_metrics(test_df.is_target, fm_preds[:, 1] > 0.5)
    metrics['window_size'] = window_size
    metrics['embed_dim'] = embed_dim
    metrics['embed_type'] = embed_type
    metrics['model'] = str(model.__class__.__name__)
    pprint(metrics)
    return metrics


if __name__ == """__main__""":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-search',
                        default=False,
                        action='store_true')
    parser.add_argument('--window-sizes',
                        default='7',
                        help='comma sep list of integer window sizes to run')
    parser.add_argument('--embed-types',
                        default='glove',
                        help='one of: glove, general_with_all_cyber, all_cyber, labeled_cyber, general')
    parser.add_argument('--embed-sizes',
                        default='100',
                        help='comma sep list of integer embed sizes to run')
    parser.add_argument('--eval-holdout',
                        default=False, action='store_true',
                        help='Train on both CV and train data, then test holdout performance')
    parser.add_argument('--target-entity',
                        default='Entity',
                        help='Target entity name: Entity, Action, Modifier')
    parser.add_argument('--model-type',
                        default='dt')
    parser.add_argument('--metrics-output-path',
                        default='auto')

    args = parser.parse_args()

    model_map = dict(
        dt=DecisionTreeClassifier(),
        rf=RandomForestClassifier(),
        gb=GradientBoostingClassifier(),
        nb=MultinomialNB()
    )


    if args.grid_search:
        basic_grid_search()
    else:

        #embed_type = 'w2v'
        embed_types = [v.strip() for v in args.embed_types.split(',')]
        win_sizes = [int(v.strip()) for v in args.window_sizes.split(',')]
        embed_sizes = [int(v.strip()) for v in args.embed_sizes.split(',')]
        models = [model_map.get(v.strip()) for v in args.model_type.split(',')]
        results = [eval_on_holdout(target_label=args.target_entity,
                                   window_size=ws, embed_dim=es, embed_type=et,
                                   model=m, eval_holdout=args.eval_holdout)
                    for m in models for ws in win_sizes for es in embed_sizes for et in embed_types]

        df = pd.DataFrame(results)
        print(df)
        if args.metrics_output_path == 'auto':
            win_sizes = "-".join(str(ws) for ws in win_sizes)
            embeds = "-".join(str(es) for es in embed_sizes)
            if args.eval_holdout:
                df.to_csv('win_size_%s_sweep_%s_%s_%s_%s_holdout.csv' % (win_sizes, args.embed_types, embeds,
                                                                            args.model_type, args.target_entity))
            else:
                df.to_csv('win_size_%s_sweep_%s_%s_%s_%s.csv' % (win_sizes, args.embed_types, embeds,
                                                                    args.model_type, args.target_entity))










