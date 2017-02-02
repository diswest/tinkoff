import argparse
import numpy as np
import pandas as pd
import re

SEED = 42
np.random.seed(SEED)

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from scipy.stats import expon
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import cv, DMatrix, XGBClassifier

N_JOBS = 7


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
    df['monthly_payment'] = df['monthly_payment'].replace(np.inf, np.nan, ).fillna(df['monthly_payment'].median())
    df['monthly_payment_to_income'] = df['monthly_payment'] / df['monthly_income']
    df['monthly_payment_to_income'] = df['monthly_payment_to_income'].replace(np.inf, np.nan).fillna(df['monthly_payment_to_income'].median())
    df['credit_sum_to_income'] = df['credit_sum'] / df['monthly_income']
    df['credit_sum_to_income'] = df['credit_sum_to_income'].replace(np.inf, np.nan).fillna(df['credit_sum_to_income'].median())

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
    print('Fit...')
    clf.fit(X_train, y_train)

    print('CV...')
    train_pred = clf.predict(X_train)
    train_predprob = clf.predict_proba(X_train)[:, 1]

    cv_score = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc', verbose=1)

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

def tune(clf, param_test, X_train, y_train):
    scv = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_test,
        n_iter=60,
        scoring='roc_auc',
        n_jobs=N_JOBS,
        iid=False,
        cv=3,
        verbose=2
    )
    scv.fit(X_train.loc[:40000].as_matrix(), y_train.loc[:40000].as_matrix())
    print(scv.best_params_)
    print(scv.best_score_)


def cv_xgb(clf, X_train, y_train):
    print('Select XGBoost n_estimators value...')
    xgb_param = clf.get_xgb_params()
    xgtrain = DMatrix(X_train, label=y_train)
    cvresult = cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=5,
                  metrics='auc', early_stopping_rounds=50, verbose_eval=True)
    clf.set_params(n_estimators=cvresult.shape[0])
    print('Optimal number of trees: %s' % cvresult.shape[0])

    return clf

    # Optimal number of trees: 3861 for learning rate 0.01 and CV AUC is 0.768535
    # Optimal number of trees: 738 for learning rate 0.05 and CV AUC is 0.768169 // Better on the test set


def xgb(X_train, y_train, X_test):
    print('XGBoost:')

    # clf = XGBClassifier(
    #     learning_rate=0.05,
    #     n_estimators=738,
    #     max_depth=5,
    #     subsample=0.9,
    #     colsample_bytree=0.6,
    #     seed=SEED
    # )

    clf = XGBClassifier(
        learning_rate=0.05,
        n_estimators=1000,  # 338(0.765053)
        max_depth=4,
        subsample=0.5,
        colsample_bytree=0.8,
        seed=SEED
    )

    fit(clf, X_train, y_train)
    print('Predict...')
    pred = clf.predict_proba(X_test)

    return pred[:, 1]


def tune_xgb(X_train, y_train):
    print('Tune XGBoost...')

    param_test = {
        'max_depth': range(1, 6),
        'subsample': [i/10 for i in range(1, 10)],
        'colsample_bytree': [i/10 for i in range(1, 10)],
        'min_child_weight': range(1, 6),
    }
    clf = XGBClassifier(
        learning_rate=0.05,
        seed=SEED
    )

    tune(clf, param_test, X_train, y_train)


def create_nn():
    model = Sequential()
    model.add(Dense(155, input_dim=174, init='uniform', activation='relu'))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])

    return model


def nn(X_train, y_train, X_test):
    print('Keras:')

    estimators = [
        ('standardize', StandardScaler()),
        ('nn', KerasClassifier(build_fn=create_nn, nb_epoch=10, batch_size=60, verbose=1))
    ]
    clf = Pipeline(estimators)

    fit(clf, X_train.as_matrix(), y_train.as_matrix())
    print('Predict...')
    pred = clf.predict_proba(X_test.as_matrix())

    return pred[:, 1]


def tune_nn(X_train, y_train):
    print('Tune Keras...')

    param_test = {
        'nn__nb_epoch': range(10, 100),
        'nn__batch_size': range(10, 100),
        'nn__loss': ['mse', 'mae', 'mape', 'msle', 'squared_hinge', 'hinge', 'binary_crossentropy', 'kld', 'poisson',
                     'cosine_proximity'],
        'nn__neurons': range(10, 200),
    }
    estimators = [
        ('standardize', StandardScaler()),
        ('nn', KerasClassifier(build_fn=create_nn, verbose=0))
    ]
    clf = Pipeline(estimators)

    tune(clf, param_test, X_train, y_train)


def knn(X_train, y_train, X_test):
    print('KNN:')

    estimators = [
        ('standardize', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=N_JOBS))
    ]
    clf = Pipeline(estimators)

    fit(clf, X_train, y_train)
    print('Predict...')
    pred = clf.predict_proba(X_test)

    return pred[:, 1]


def tune_knn(X_train, y_train):
    print('Tune KNN...')

    param_test = {
        'knn__n_neighbors': range(2, 20),
        'knn__weights': ['uniform', 'distance'],
        'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'knn__leaf_size': range(10, 100)
    }
    estimators = [
        ('standardize', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=N_JOBS))
    ]
    clf = Pipeline(estimators)

    tune(clf, param_test, X_train, y_train)


def rf(X_train, y_train, X_test):
    print('RF:')

    clf = RandomForestClassifier(
        n_jobs=N_JOBS,
        random_state=SEED
    )

    fit(clf, X_train, y_train)
    print('Predict...')
    pred = clf.predict_proba(X_test)

    return pred[:, 1]


def tune_rf(X_train, y_train):
    print('Tune RF...')
    param_test = {
        'n_estimators': range(1, 300),
        'max_features': ['auto', 'sqrt', 'log2', None],
        'max_depth': range(1, 10),
        'min_samples_split': range(2, 20),
        'min_samples_leaf': range(1, 20)
    }

    clf = RandomForestClassifier(
        n_jobs=N_JOBS,
        random_state=SEED
    )

    tune(clf, param_test, X_train, y_train)


def lr(X_train, y_train, X_test):
    print('LR:')

    estimators = [
        ('standardize', StandardScaler()),
        ('lr', LogisticRegression(C=0.0779, solver='sag', n_jobs=N_JOBS, random_state=SEED))
    ]
    clf = Pipeline(estimators)

    fit(clf, X_train, y_train)
    print('Predict...')
    pred = clf.predict_proba(X_test)

    return pred[:, 1]


def tune_lr(X_train, y_train):
    print('Tune LR...')
    param_test = {
        'lr__C': expon(scale=100),
        'lr__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag']
    }
    estimators = [
        ('standardize', StandardScaler()),
        ('lr', LogisticRegression(n_jobs=N_JOBS, random_state=SEED))
    ]
    clf = Pipeline(estimators)

    tune(clf, param_test, X_train, y_train)


def svm(X_train, y_train, X_test):
    print('SVM:')

    estimators = [
        ('standardize', StandardScaler()),
        ('svm', SVC(random_state=SEED))
    ]
    clf = Pipeline(estimators)

    fit(clf, X_train, y_train)
    print('Predict...')
    pred = clf.predict_proba(X_test)

    return pred[:, 1]


def tune_svm(X_train, y_train):
    print('Tuning SVM...')
    param_test = {
        'svm__C': expon(scale=100),
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'svm__degree': range(1, 10),
        'svm__gamma': expon(scale=.1),
        'svm__class_weight': ['balanced', None]
    }
    estimators = [
        ('standardize', StandardScaler()),
        ('svm', SVC(random_state=SEED))
    ]
    clf = Pipeline(estimators)

    tune(clf, param_test, X_train, y_train)


def run(submission_name):
    l1_models_pool = {
        'xgb': xgb,
        'nn': nn,
        'knn': knn,
        'rf': rf,
        'lr': lr,
        'svm': svm
    }

    print('Prepare datasets...')
    train, X_test = prepare()

    y_train = train['open_account_flg']
    X_train = train.drop('open_account_flg', axis=1)


    # tune_svm(X_train, y_train)
    # for i in range(5):
    #     print('#' * 80)
    # tune_knn(X_train, y_train)
    # for i in range(5):
    #     print('#' * 80)
    # tune_rf(X_train, y_train)
    # return


    pred = xgb(X_train, y_train, X_test)

    print('Build submission...')
    df = pd.read_csv('data/credit_test.csv', sep=';')
    submission = pd.DataFrame({
        '_ID_': df['client_id'],
        '_VAL_': pred
    })

    print('Write submission to file (%s.csv)...' % submission_name)
    submission.to_csv('submissions/%s.csv' % submission_name, index=False)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('submission')
    args = parser.parse_args()

    run(args.submission)


#### VOTING CLASSIFIER


# Сделать фичи из отношения зарплаты и размера кредита к среднерегиональной зарплате
# Сделать фичу из ежемесячного платежа
# Сделать фичу из отношения ежемесячного платежа к зарплате
# Попробовать угадывать регион по зарплате и работе