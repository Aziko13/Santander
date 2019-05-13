import os
import shutil
import pandas as pd
import src.classLGB as classLGB
import gc
import numpy as np
from tqdm import tqdm


random_state = 42
nrows=100000
NFOLDS = 8
np.random.seed(random_state)


script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)

print("Model: {}".format(MODEL_NAME))
print("Reading training data")

np.random.seed(random_state)

# LOADING DATA SETS

# data_path = r'/home/aziz/Desktop/Santander Customer Transaction Prediction/Data/'
data_path = r'C:/Users/a_abdraimov/Desktop/Docs/Compet/Santander Customer Transaction Predition/input/'

df_train = pd.read_csv(data_path+'train.csv', nrows=nrows)
df_test = pd.read_csv(data_path+'test.csv', nrows=nrows)
gc.collect()


# IDENTIFYING SYNTHETIC OBSERATIONS

def get_synthetic_spas(df):

    df = df.drop(['ID_code'], axis=1, inplace=False)
    df = df.values

    unique_samples = []
    unique_count = np.zeros_like(df)
    for feature in tqdm(range(df.shape[1])):
        _, index_, count_ = np.unique(df[:, feature], return_counts=True, return_index=True)
        unique_count[index_[count_ == 1], feature] += 1

    # Samples which have unique values are real the others are fake
    real_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]
    synthetic_samples_indexes = np.argwhere(np.sum(unique_count, axis=1) == 0)[:, 0]

    return synthetic_samples_indexes, real_samples_indexes

synthetic_samples_indexes, real_samples_indexes = get_synthetic_spas(df_test)

print(df_test.shape, df_train.shape)

gc.collect()
FEATS_EXCLUDED = ['ID_code', 'index', 'target']
target_col = 'target'
predictor_cols = [f for f in df_train.columns if f not in FEATS_EXCLUDED]


# ADDITIONAL FEATURES
# Creating frequency features for all predictors

df = pd.concat([df_train, df_test.loc[real_samples_indexes]], axis=0, sort=False)

for f in tqdm(predictor_cols[0:20]):
    cc = np.unique(df[f], return_counts=True)
    cc = pd.DataFrame({f: cc[0], f + '_cnts': cc[1]})

    df_train = df_train.merge(cc, on=f, how='left')
    df_test = df_test.merge(cc, on=f, how='left')

gc.collect()

# ADDITIONAL FEATURES

def add_feats(df):
    df['cnts_sum'] = np.sum(df[[x for x in df.columns if x.find('_cnts') > 0]], axis=1)
    df['cnts_min'] = np.min(df[[x for x in df.columns if x.find('_cnts') > 0]], axis=1)
    df['cnts_max'] = np.max(df[[x for x in df.columns if x.find('_cnts') > 0]], axis=1)
    df['cnts_avg'] = np.mean(df[[x for x in df.columns if x.find('_cnts') > 0]], axis=1)
    df['cnts_std'] = np.std(df[[x for x in df.columns if x.find('_cnts') > 0]], axis=1)
    df['cnts_1_cnt'] = np.sum(df[[x for x in df.columns if x.find('_cnts') > 0]] == 1, axis=1)

    gc.collect()
    return (df)

df_train = add_feats(df_train)
df_test = add_feats(df_test)

print(df_test.shape, df_train.shape)

predictor_cols = [f for f in df_train.columns if f not in FEATS_EXCLUDED]

print("Num of predictors is {0}".format(len(predictor_cols)))


# MODELING     CREATING LGB MODEL

params = {
    'boost': 'gbdt',
    'metric': 'auc',
    'objective': 'binary',
    'verbosity': 1,
    'num_threads': -1,
    'seed': random_state,

    'tree_learner': 'serial',
    'learning_rate': 0.01,
    'num_leaves': 3,
    'max_depth': -1,
    'min_data_in_leaf': 40,
    'bagging_fraction': 0.3,
    'bagging_freq': 0,
    'feature_fraction': 1.0
}

lgb_my = classLGB.classLGB(df_train, df_test, target_col, predictor_cols, id_col='ID_code')


# nrounds = 100000
nrounds = 10
early_stopping = 1000

y_preds, model, best_iters, best_scores, feature_importance_df, oof_preds = lgb_my.kfold_train(NFOLDS,
                                                                                               params=params,
                                                                                               random_state=12,
                                                                                               shuffle=True,
                                                                                               nrounds=nrounds,
                                                                                               early_stopping=early_stopping,
                                                                                               oversampling=None)



print('-'*100)
print(round(np.mean(best_scores),5))
print('-'*100)

print("Saving OOF predictions")

print("Saving OOF predictions")
oof_preds = pd.DataFrame(np.column_stack((df_train['ID_code'], oof_preds.ravel())), columns=['ID_code', 'target'])
oof_preds.to_csv('../kfolds/{}__{}.csv'.format(MODEL_NAME, str(round(np.mean(best_scores),5))), index=False)

print("Saving code to reproduce")
shutil.copyfile(os.path.basename(__file__),'../model_source/{}__{}.py'.format(MODEL_NAME, str(round(np.mean(best_scores),5))))

print("Saving submission file")
y_preds['target'] = y_preds.iloc[:, 1:].apply(np.mean, axis=1)
y_preds.rename(columns={'id':'ID_code'}, inplace=True)
y_preds[['ID_code', 'target']].to_csv('../model_predictions/submission_{}__{}.csv'.format(MODEL_NAME,str(round(np.mean(best_scores),5))), index=False)


