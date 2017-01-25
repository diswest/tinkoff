import pandas as pd
import xgboost as xgb
import re

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, KFold



def clean_dataset(df, dropna=True):
    df = df.copy()

    # Fix data
    df['credit_sum'] = df['credit_sum'].apply(lambda x: str(x).replace(',', '.')).astype(float)
    # Hanssen and Kuipers discriminant
    df['score_shk'] = df['score_shk'].apply(lambda x: str(x).replace(',', '.')).astype(float)

    # Clean data
    df.drop('client_id', axis=1, inplace=True)
    df['monthly_income'].fillna(df['monthly_income'].mean(), inplace=True)
    df['credit_count'].fillna(df['credit_count'].mean(), inplace=True)
    df['overdue_credit_count'].fillna(df['overdue_credit_count'].mean(), inplace=True)
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

    #df['living_region'] = df['living_region'].apply(clean_region)
    df.drop('living_region', axis=1, inplace=True)

    return df


def prepare():
    train = clean_dataset(pd.read_csv('data/credit_train.csv', sep=';'))
    test = clean_dataset(pd.read_csv('data/credit_test.csv', sep=';'), dropna=False)

    # Convert categorical features
    #columns = ['gender', 'marital_status', 'job_position', 'tariff_id', 'education', 'living_region']
    columns = ['gender', 'marital_status', 'job_position', 'tariff_id', 'education']
    for col in columns:
        categories = train[col].copy().append(test[col].copy()).unique()
        train[col] = train[col].astype('category', categories=categories)
        test[col] = test[col].astype('category', categories=categories)
    return pd.get_dummies(train, columns=columns), pd.get_dummies(test, columns=columns)


print('Prepare datasets...')
train, X_test = prepare()
y_train = train['open_account_flg']
X_train = train.drop('open_account_flg', axis=1)
print(X_test.head(1))

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05)

print('Check...')
kf = KFold(5, random_state=42)
scores = cross_val_score(gbm, X_train, y_train, cv=kf, scoring='roc_auc')
print('Result: %s' % scores.mean())

print('Fit...')
gbm.fit(X_train, y_train)
print('Predict...')
pred = gbm.predict(X_test)

df = pd.read_csv('data/credit_test.csv', sep=';')
submission = pd.DataFrame({
    '_ID_': df['client_id'],
    '_VAL_': pred
})

print('Submission [%s]:' % submission.shape[0])
print(submission.head())
print('...')
print(submission.tail())
print('Write submission to file...')
submission.to_csv('submission.csv', index=False)
print('Done!')
