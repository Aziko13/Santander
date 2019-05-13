import numpy as np
import pandas as pd
import os

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import gc

from tqdm import tqdm



# ---------- INPUT VARS ---------
random_state = 42
nrows=None
data_path = r"C:\\Users\\a_abdraimov\\Desktop\\Docs\\Compet\\Santander Customer Transaction Predition\\input\\"

# ------------------------------

np.random.seed(random_state)
df_train = pd.read_csv(data_path+'train.csv', nrows=nrows)
df_test = pd.read_csv(data_path+'test.csv', nrows=nrows)

df = pd.concat([df_train, df_test], axis=0)

FEATS_EXCLUDED = ['ID_code', 'index', 'target']
target_col = 'target'
predictor_cols = [x for x in df.columns if x not in FEATS_EXCLUDED]

# predictor_cols = predictor_cols[0:10]

# ----------- CNTS VARS for all features -----
for f in tqdm(predictor_cols):
    cc = np.unique(df[f], return_counts=True)
    cc = pd.DataFrame({f: cc[0], f + '_cnts': cc[1]})
    df = df.merge(cc, on=f)

gc.collect()

df['cnts_sum'] = np.sum(df[[x for x in df.columns if x.find('_cnts')>0]], axis=1)
df['cnts_min'] = np.min(df[[x for x in df.columns if x.find('_cnts')>0]], axis=1)
df['cnts_max'] = np.max(df[[x for x in df.columns if x.find('_cnts')>0]], axis=1)
df['cnts_avg'] = np.mean(df[[x for x in df.columns if x.find('_cnts')>0]], axis=1)
df['cnts_std'] = np.std(df[[x for x in df.columns if x.find('_cnts')>0]], axis=1)
df['cnts_1_cnt'] = np.sum(df[[x for x in df.columns if x.find('_cnts')>0]]==1, axis=1)

gc.collect()

predictor_cols = [f for f in df.columns if f not in FEATS_EXCLUDED]

print("Predictors len is {0}".format(len(predictor_cols)))

df_train = df[df['target'].notnull()]
df_test = df[df['target'].isnull()]
del df

X_dev, X_val, y_dev, y_val = train_test_split(df_train[predictor_cols], df_train[target_col],
                                              random_state=random_state, stratify=df_train[target_col])

del df_train, df_test

f = open("test.txt", 'a')
f.writelines("New params tuning \n")
f.close()

params = {
    'boost': 'gbdt',
    'metric': 'auc',
    'objective': 'binary',
    'verbosity': 1,
    'num_threads': 4,
    'seed': random_state,

    'tree_learner': 'serial',
    'learning_rate': 0.03,
    'num_leaves': 10,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'bagging_fraction': 1.0,
    'bagging_freq': 0,
    'feature_fraction': 1.0
}

num_leaves_l = [5, 10, 13]
min_data_in_leaf_l = [10, 20, 30]
bagging_fraction_l = [0.3, 0.7, 1.0]
bagging_freq_l = [0, 3, 10]
feature_fraction_l = [0.3, 0.7, 1.0]
# num_leaves_l = [5, 10]
# min_data_in_leaf_l = [10]
# bagging_fraction_l = [0.3]
# bagging_freq_l = [0]
# feature_fraction_l = [0.3]

print("Running cycles")

for num_leaves in tqdm(num_leaves_l):
    for min_data_in_leaf in min_data_in_leaf_l:
        for bagging_fraction in bagging_fraction_l:
            for bagging_freq in bagging_freq_l:
                for feature_fraction in feature_fraction_l:
                    params['num_leaves'] = num_leaves
                    params['min_data_in_leaf'] = min_data_in_leaf
                    params['bagging_fraction'] = bagging_fraction
                    params['bagging_freq'] = bagging_freq
                    params['feature_fraction'] = feature_fraction

                    evals_result = {}
                    lgtrain = lgb.Dataset(X_dev, label=y_dev)
                    lgval = lgb.Dataset(X_val, label=y_val)

                    model = lgb.train(params,
                                      lgtrain,
                                      num_boost_round=100000,
                                      # num_boost_round=10,
                                      valid_sets=[lgval],
                                      early_stopping_rounds=1000,
                                      verbose_eval=-1,
                                      evals_result=evals_result)

                    val_predict = model.predict(X_val)
                    score = roc_auc_score(y_val, val_predict)
                    gc.collect()

                    res = str(num_leaves) + "\t" + str(min_data_in_leaf) + "\t" + str(bagging_fraction) + "\t" + str(
                        bagging_freq) + "\t" + str(feature_fraction) + "\t" + str(score)
                    f = open("test.txt", 'a')
                    f.writelines(res + "\n")
                    f.close()
