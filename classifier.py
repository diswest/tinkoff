import argparse
import numpy as np
import os
import pandas as pd
import re


SEED = 42
np.random.seed(SEED)

from keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier

from lightgbm import LGBMClassifier

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import cv, DMatrix, XGBClassifier

N_JOBS = 1
os.environ['OMP_NUM_THREADS'] = '%s' % 8
# os.environ['JOBLIB_TEMP_FOLDER'] = '/notebooks/tmp'


def clean_dataset(df, dropna=True):
    df = df.copy()

    # Fix data
    df['credit_sum'] = df['credit_sum'].apply(lambda x: str(x).replace(',', '.')).astype(float)
    # Hanssen and Kuipers discriminant
    df['score_shk'] = df['score_shk'].apply(lambda x: str(x).replace(',', '.')).astype(float)

    # Clean data
    df.drop('client_id', axis=1, inplace=True)
    df['monthly_income'].replace(0, np.nan, inplace=True)
    df['monthly_income'].fillna(df['monthly_income'].mean(), inplace=True)
    df['credit_count'].fillna(df['credit_count'].mean(), inplace=True)
    df['overdue_credit_count'].fillna(df['overdue_credit_count'].mean(), inplace=True)
    if dropna:
        df.dropna(axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)

    def clean_region(region):
        if type(region) != str:
            return np.nan
        patterns = [
            '(\W|^)А?ОБЛ(\W|$)', 'ОБЛАСТЬ',
            '(\W|^)Г(\W|$)', '(\W|^)ГОРОД(\W|$)',
            '(\W|^)РЕСП(\W|$)', 'РЕСПУБЛИКА',
            '(\W|^)КР(\W|$)', 'КРАЙ',
            '(\W|^)АО(\W|$)', 'АВТОНОМНЫЙ ОКРУГ', 'АВТОНОМНАЯ', 'АO',
            'РАЙОН', 'Р-Н',
            'ФЕДЕРАЛЬНЫЙ ОКРУГ'
        ]
        regions_pattern = '(%s)' % '|'.join(patterns)
        result = re.sub(regions_pattern, '', region.upper())
        result = re.sub('\W', '', result)
        result = re.sub('АЯ(\W|$)|ИЙ(\W|$)', '', result)
        result = re.sub('.*ЧУВАШ.*', 'ЧУВАШ', result)
        return result

    df['living_region'] = df['living_region'].apply(clean_region)
    df['living_region'].fillna(df['living_region'].mode()[0], inplace=True)

    # Add metafeatures
    df['monthly_payment'] = df['credit_sum'] / df['credit_month']
    df['monthly_payment_to_income'] = df['monthly_payment'] / df['monthly_income']
    df['credit_sum_to_income'] = df['credit_sum'] / df['monthly_income']

    return df


def prepare():
    train = clean_dataset(pd.read_csv('data/credit_train.csv', sep=';'))
    test = clean_dataset(pd.read_csv('data/credit_test.csv', sep=';'), dropna=False)

    # Convert categorical features
    columns = ['gender', 'marital_status', 'job_position', 'tariff_id', 'education', 'living_region']
    for col in columns:
        categories = train[col].copy().append(test[col]).unique()
        train[col] = train[col].astype('category', categories=categories)
        test[col] = test[col].astype('category', categories=categories)
    return pd.get_dummies(train, columns=columns), pd.get_dummies(test, columns=columns)


def prepare_dropout():
    train = clean_dataset(pd.read_csv('data/credit_train.csv', sep=';'))
    test = clean_dataset(pd.read_csv('data/credit_test.csv', sep=';'), dropna=False)

    result = {}

    for skip_col in train.columns:
        if skip_col == 'open_account_flg':
            continue
        tr = train.drop(skip_col, axis=1)
        ts = test.drop(skip_col, axis=1)

        # Convert categorical features
        columns = ['gender', 'marital_status', 'job_position', 'tariff_id', 'education', 'living_region']
        if skip_col in columns:
            columns.remove(skip_col)

        for col in columns:
            categories = tr[col].copy().append(ts[col].copy()).unique()
            tr[col] = tr[col].astype('category', categories=categories)
            ts[col] = ts[col].astype('category', categories=categories)
        result[skip_col] = (pd.get_dummies(tr, columns=columns), pd.get_dummies(ts, columns=columns))
    return result


def fit(clf, X_train, y_train):
    print('Fit...')
    clf.fit(X_train, y_train)

    print('CV...')
    cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc', verbose=2, n_jobs=N_JOBS)

    train_pred = clf.predict(X_train)
    train_predprob = clf.predict_proba(X_train)[:, 1]

    if type(y_train) == pd.DataFrame:
        print('Accuracy: %f' % accuracy_score(y_train.values, train_pred))
    else:
        print('Accuracy: %f' % accuracy_score(y_train, train_pred))

    print('AUC Score: %f' % roc_auc_score(y_train, train_predprob))
    print('CV Score: min - %f, max - %f, mean - %f, std – %f' % (
        np.min(cv_score),
        np.max(cv_score),
        np.mean(cv_score),
        np.std(cv_score)
    ))

    return np.mean(cv_score)


def tune(clf, param_test, X_train, y_train, n=60):
    scv = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_test,
        n_iter=n,
        scoring='roc_auc',
        n_jobs=N_JOBS,
        iid=False,
        cv=3,
        verbose=2
    )
    scv.fit(X_train.as_matrix(), y_train.as_matrix())
    print(scv.best_params_)
    print(scv.best_score_)


def fit_stacking(l1_models_pool, l2_model, X_train, y_train, X_test, l1_features=None):
    print()
    print('Stacking:')

    print('Generate L1 train metafeatures:')

    l1_df_train = pd.DataFrame(index=X_train.index)
    l1_models_to_fit = {}
    for name, model in l1_models_pool.items():
        pred = load_l1_predictions('%s_folded' % name)
        if pred is None:
            l1_models_to_fit[name] = model
        else:
            l1_df_train[name] = pred

    if l1_models_to_fit:
        fit_stacking_l1_folded(l1_models_to_fit, X_train, y_train, l1_features)
        pred = predict_stacking_l1(l1_models_to_fit, X_train, l1_features)
        l1_df_train.update(pred)
        for name, data in pred.items():
            save_l1_predictions('%s_folded' % name, data)

    l1_df_train.dropna(inplace=True)

    print('Generate L1 test metafeatures:')

    l1_df_test = pd.DataFrame(index=X_test.index)
    l1_models_to_fit = {}
    for name, model in l1_models_pool.items():
        pred = load_l1_predictions(name)
        if pred is None:
            l1_models_to_fit[name] = model
        else:
            l1_df_test[name] = pred

    if l1_models_to_fit:
        fit_stacking_l1(l1_models_to_fit, X_train, y_train, l1_features)
        pred = predict_stacking_l1(l1_models_to_fit, X_test, l1_features)
        l1_df_test.update(pred)
        for name, data in pred.items():
            save_l1_predictions('%s' % name, data)

    print('Fit L2 model...')
    return l1_df_test, fit(l2_model, l1_df_train, y_train)


def fit_stacking_l1(l1_models_pool, X, y, features=None):
    print('Fit L1 models:')
    if not features:
        features = {}

    model_idx = 0

    for name, clf in l1_models_pool.items():
        model_idx += 1
        print('[%s/%s] %s...' % (model_idx, len(l1_models_pool), name))

        predictors = features[name] if name in features else X.columns
        clf.fit(X[predictors], y)


def fit_stacking_l1_folded(l1_models_pool, X, y, features=None):
    print('Fit L1 models (folded):')
    if not features:
        features = {}

    seed_offset = 0
    model_idx = 0

    l1_df = pd.DataFrame(index=y.index)

    for name, clf in l1_models_pool.items():
        model_idx += 1
        print('[%s/%s] %s:' % (model_idx, len(l1_models_pool), name))
        l1_df[name] = np.nan
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+seed_offset)
        fold_idx = 0
        folds_cv = []
        for train, test in kf.split(X, y):
            fold_idx += 1
            print('[%s/%s] fold %s...' % (model_idx, len(l1_models_pool), fold_idx))

            predictors = features[name] if name in features else X.columns
            X_train, y_train = X[predictors].loc[train], y.loc[train]
            X_test, y_test = X[predictors].loc[test], y.loc[test]

            clf.fit(X_train, y_train)

            pred_proba = clf.predict_proba(X_test)[:, 1]
            folds_cv.append(roc_auc_score(y_test, pred_proba))
            pred_df = pd.DataFrame({name: pred_proba}, index=y_test.index)
            l1_df.update(pred_df)
        print('[%s/%s] CV AUC Score: min: %f, max: %f, mean: %f, std: %f' % (
            model_idx,
            len(l1_models_pool),
            np.min(folds_cv),
            np.max(folds_cv),
            np.mean(folds_cv),
            np.std(folds_cv),
        ))

        seed_offset += 1

    return l1_df


def predict_stacking_l1(l1_models_pool, X, features=None):
    print('Predict L1 models...')
    if not features:
        features = {}

    model_idx = 0

    l1_df = pd.DataFrame()

    for name, clf in l1_models_pool.items():
        model_idx += 1
        print('[%s/%s] %s...' % (model_idx, len(l1_models_pool), name))

        l1_df[name] = np.nan

        predictors = features[name] if name in features else X.columns
        pred_proba = clf.predict_proba(X[predictors])[:, 1]

        l1_df[name] = pred_proba

    return l1_df


def load_l1_predictions(name):
    fname = 'models/%s.csv' % name
    if os.path.isfile(fname):
        print('Load %s from cache' % name)
        return pd.read_csv(fname, header=None)
    else:
        return None


def save_l1_predictions(name, df):
    fname = 'models/%s.csv' % name
    print('Save %s to cache' % name)
    df.to_csv(fname, index=False)


def xgb():
    return XGBClassifier(
        learning_rate=0.05,
        n_estimators=738,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.6,
        nthread=4,
        seed=SEED
    )


def xgb_cv(clf, X_train, y_train):
    print('Select XGBoost n_estimators value...')
    xgb_param = clf.get_xgb_params()
    xgtrain = DMatrix(X_train, label=y_train)
    cvresult = cv(
        xgb_param,
        xgtrain,
        num_boost_round=clf.get_params()['n_estimators'],
        nfold=5,
        metrics='auc',
        early_stopping_rounds=50,
        verbose_eval=True
    )
    clf.set_params(n_estimators=cvresult.shape[0])
    print('Optimal number of trees: %s' % cvresult.shape[0])

    return clf


def xgb_bag():
    return BaggingClassifier(
        base_estimator=xgb(),
        n_estimators=14,
        random_state=SEED
    )


def xgb_bag_calibrated():
    return CalibratedClassifierCV(xgb_bag(), cv=5)


def create_nn():
    model = Sequential()
    model.add(Dense(155, input_dim=71, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])

    return model


def nn():
    estimators = [
        ('standardize', StandardScaler()),
        ('nn', KerasClassifier(build_fn=create_nn, nb_epoch=10, batch_size=60, verbose=0))
    ]

    return Pipeline(estimators)


def nn_bag():
    return BaggingClassifier(
        nn(),
        n_estimators=10,
        random_state=SEED
    )


def rf():
    return RandomForestClassifier(
        n_estimators=252,
        min_samples_split=6,
        min_samples_leaf=5,
        max_features='sqrt',
        max_depth=9,
        n_jobs=N_JOBS,
        random_state=SEED
    )


def gbm():
    return LGBMClassifier(
        learning_rate=0.05,
        scale_pos_weight=2,
        num_leaves=30,
        n_estimators=480,
        min_child_weight=1,
        min_child_samples=12,
        max_bin=110,
        colsample_bytree=0.3,
        seed=SEED
    )


def gbm_bag():
    return BaggingClassifier(
        base_estimator=gbm(),
        n_estimators=19,
        random_state=SEED
    )


def calibrated(clf):
    return CalibratedClassifierCV(clf, cv=5)


def run(submission_name):
    print('Prepare datasets...')
    train, X_test = prepare()
    y_train = train['open_account_flg']
    X_train = train.drop('open_account_flg', axis=1)

    #  Part 1. Boosting. (Part 2 must be commented)

    l1_features = {
        'xgb': [feature for feature in X_train.columns
                if feature not in ['monthly_payment', 'monthly_payment_to_income', 'credit_sum_to_income']],
        'nn': [feature for feature in X_train.columns
               if feature not in ['credit_sum_to_income'] and feature[:13] != 'living_region'],
        'rf': [feature for feature in X_train.columns
               if feature[:13] != 'living_region'],
        'gbm': [feature for feature in X_train.columns
                if feature not in ['monthly_income', 'monthly_payment_to_income']]
    }

    l1_models_pool = {
        'xgb': calibrated(xgb_bag()),
        # 'nn': calibrated(nn_bag()),
        # 'rf': calibrated(rf()),
        'gbm': calibrated(gbm_bag())
    }
    l2_model = LogisticRegressionCV(
        cv=5,
        scoring='roc_auc',
        max_iter=10000,
        solver='sag',
        class_weight='balanced',
        n_jobs=N_JOBS,
        random_state=SEED
    )

    l1_df, cv_score = fit_stacking(
        l1_models_pool,
        l2_model,
        X_train, y_train,
        X_test,
        l1_features=l1_features
    )

    print()
    print('Predict...')
    pred = l2_model.predict_proba(l1_df)[:, 1]

    # End part 1

    # Part 2. Over previous submissions (Part 1 must be commented)

    df = pd.DataFrame({
        'stacked': pd.read_csv('submissions/stacking_xgb_gbm_calibrated_l2_lrcv_calibrated_0.79973117848.csv', index_col='_ID_')['_VAL_'].as_matrix(),
        'xgb': load_l1_predictions('xgb')[0],
        'gbm': load_l1_predictions('gbm')[0]
    })

    pred = df.mean(axis=1)

    cv_score = '___'

    # End part 2

    print()
    print('Build submission...')
    df = pd.read_csv('data/credit_test.csv', sep=';')
    submission = pd.DataFrame({
        '_ID_': df['client_id'],
        '_VAL_': pred
    })
    print('Write submission to file (%s.csv)...' % submission_name)
    submission.to_csv('submissions/%s_%s.csv' % (submission_name, cv_score), index=False)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('submission')
    args = parser.parse_args()

    run(args.submission)
