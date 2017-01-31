import numpy as np
import pandas as pd
import re

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import cv, DMatrix
from xgboost.sklearn import XGBClassifier


def clean_dataset(df, dropna=True):
    df = df.copy()

    # Fix data
    df['credit_sum'] = df['credit_sum'].apply(lambda x: str(x).replace(',', '.')).astype(float)
    # Hanssen and Kuipers discriminant
    df['score_shk'] = df['score_shk'].apply(lambda x: str(x).replace(',', '.')).astype(float)

    # Clean data
    df.drop('client_id', axis=1, inplace=True)
    df['monthly_income'].fillna(df['monthly_income'].median(), inplace=True)
    df['credit_count'].fillna(df['credit_count'].median(), inplace=True)
    df['overdue_credit_count'].fillna(df['overdue_credit_count'].median(), inplace=True)
    if dropna:
        df.dropna(axis=0, inplace=True)

    def clean_region(region):
        if type(region) != str:
            return ''
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

    return df


def prepare():
    train = clean_dataset(pd.read_csv('data/credit_train.csv', sep=';'))
    test = clean_dataset(pd.read_csv('data/credit_test.csv', sep=';'), dropna=False)

    # Convert categorical features
    columns = ['gender', 'marital_status', 'job_position', 'tariff_id', 'education', 'living_region']
    for col in columns:
        categories = train[col].copy().append(test[col].copy()).unique()
        train[col] = train[col].astype('category', categories=categories)
        test[col] = test[col].astype('category', categories=categories)
    return pd.get_dummies(train, columns=columns), pd.get_dummies(test, columns=columns)


def fit(clf, X_train, y_train):
    print('Fit XGBoost...')
    clf.fit(X_train, y_train)

    train_pred = clf.predict(X_train)
    train_predprob = clf.predict_proba(X_train)[:,1]

    cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')

    print('Accuracy: %f' % accuracy_score(y_train.values, train_pred))
    print('AUC Score: %f' % roc_auc_score(y_train, train_predprob))
    print('CV Score: min - %f, max - %f, mean - %f' % (
        np.min(cv_score),
        np.max(cv_score),
        np.mean(cv_score)
    ))

    print('Feature importances:')
    print(pd.Series(clf.feature_importances_, list(X_train.columns)).head(10).sort_values(ascending=False))


def fit_xgb(clf, X_train, y_train):
    # print('Select XGBoost n_estimators value...')
    # xgb_param = clf.get_xgb_params()
    # xgtrain = DMatrix(X_train, label=y_train)
    # cvresult = cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=5,
    #               metrics='auc', early_stopping_rounds=50, verbose_eval=True)
    # clf.set_params(n_estimators=cvresult.shape[0])
    # print('Optimal number of trees: %s' % cvresult.shape[0])

    # Optimal number of trees: 3861 for learning rate 0.01 and CV AUC is 0.768535
    # Optimal number of trees: 738 for learning rate 0.05 and CV AUC is 0.768169

    fit(clf, X_train, y_train)

print('Prepare datasets...')
train, X_test = prepare()

y_train = train['open_account_flg']
X_train = train.drop('open_account_flg', axis=1)

xgb = XGBClassifier(
    learning_rate=0.05,
    n_estimators=5000,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.6,
    seed=42,
    silent=False
)
# rf = RandomForestClassifier(
#     n_estimators=100,
#     min_samples_leaf=50,
#     min_samples_split=20,
#     max_features=None,
#     random_state=42,
#     oob_score=True,
#     n_jobs=-1
# )

print('XGBoost...')
fit_xgb(xgb, X_train, y_train)
# print('Fit Random Forest...')
# fit(rf, X_train, y_train)

# param_test1 = {
#     'max_depth': range(3, 10, 2),
#     'min_child_weight': range(1, 6, 2)
# }

# gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.2, n_estimators=127, max_depth=5,
#                                                 subsample=0.8, colsample_bytree=0.8, seed=42),
#                         param_grid=param_test1, scoring='roc_auc', n_jobs=-1, iid=False, cv=5, verbose=1)
# gsearch1.fit(X_train, y_train)
# print(gsearch1.best_params_)
# print(gsearch1.best_score_)

# param_test2 = {
#     'gamma': [i/10.0 for i in range(0,5)]
# }
#
#
# gsearch2 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.2, n_estimators=127, max_depth=5,
#                                                 subsample=0.8, colsample_bytree=0.8, seed=42),
#                         param_grid=param_test2, scoring='roc_auc', n_jobs=-1, iid=False, cv=5, verbose=1)
# gsearch2.fit(X_train, y_train)
# print(gsearch2.best_params_)
# print(gsearch2.best_score_)

# param_test3 = {
#     'subsample': [i/10.0 for i in range(6,10)],
#     'colsample_bytree': [i/10.0 for i in range(6,10)]
# }
# gsearch3 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.2, n_estimators=127, max_depth=5,
#                                                 subsample=0.8, colsample_bytree=0.8, seed=42),
#                         param_grid=param_test3, scoring='roc_auc', n_jobs=-1, iid=False, cv=5, verbose=1)
# gsearch3.fit(X_train, y_train)
# print(gsearch3.best_params_)
# print(gsearch3.best_score_)

print('Predict XGBoost...')
xgb_pred = xgb.predict_proba(X_test)
# print('Predict Random Forest...')
# rf_pred = rf.predict_proba(X_test)

# print('Merge...')
# pred = (xgb_pred[:,1] + rf_pred[:,1]) / 2
pred = xgb_pred[:,1]

print('Build submission...')
df = pd.read_csv('data/credit_test.csv', sep=';')
submission = pd.DataFrame({
    '_ID_': df['client_id'],
    '_VAL_': pred
})

print('Write submission to file...')
submission.to_csv('submission_xgb.csv', index=False)
print('Done!')
