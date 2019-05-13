import os
import shutil
import pandas as pd
import DataManipulation as DM
import classLGB
import gc
import numpy as np
from sklearn.model_selection import StratifiedKFold


# --------------- Model Name
NFOLDS = 10
RANDOM_STATE = 42
nrows = 10000

script_name = os.path.basename(__file__).split('.')[0]
MODEL_NAME = "{0}__folds{1}".format(script_name, NFOLDS)

print("Model: {}".format(MODEL_NAME))

print("Reading training data")
df = DM.load_train_test('../input/', num_rows=nrows)

gc.collect()
FEATS_EXCLUDED = ['ID_code', 'index', 'target']
target_col = 'target'
predictor_cols = [f for f in df.columns if f not in FEATS_EXCLUDED]

# -------------- Updating data

# --- round features
df = DM.get_round_features(df, predictor_cols, dec=3)


# --- top feature bins
ffs = ['var_81_rnd_3','var_139_rnd_3','var_12_rnd_3','var_53_rnd_3','var_110_rnd_3',
       'var_26_rnd_3','var_6_rnd_3','var_146_rnd_3','var_174_rnd_3','var_22_rnd_3',
       'var_76_rnd_3','var_166_rnd_3','var_80_rnd_3','var_99_rnd_3','var_109_rnd_3',
       'var_21_rnd_3','var_165_rnd_3','var_13_rnd_3','var_133_rnd_3','var_2_rnd_3']

df = DM.get_binned_features(df, ffs, num_bins=30, postf='_30')
df = DM.get_binned_features(df, ffs, num_bins=5, postf='_5')

# --- target encoding
feats_for_te = [x for x in df.columns if '_binned_' in x]
df = DM.get_target_encoding(df, feats_for_te , replace=False, weight=300, postf='')

# ------------------ Modeling: LGB

FEATS_EXCLUDED = ['ID_code', 'index', 'target']

train_df = df[df['target'].notnull()]
test_df = df[df['target'].isnull()]

target_col = 'target'
predictor_cols = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

predictor_cols = predictor_cols[200:]


lgb_my = classLGB.classLGB(train_df, test_df, target_col, predictor_cols, id_col='ID_code')

param = {
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 2,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary',
    'verbosity': 1
}

# nrounds = 1000000
nrounds = 10
early_stopping = 3000
y_preds, model, best_iters, best_scores, feature_importance_df, oof_preds = lgb_my.kfold_train(NFOLDS, params=param, random_state=12, shuffle=True, nrounds=nrounds, early_stopping=early_stopping)


print("Saving OOF predictions")


print("Saving OOF predictions")
oof_preds = pd.DataFrame(np.column_stack((train_df['ID_code'], oof_preds.ravel())), columns=['ID_code', 'target'])
oof_preds.to_csv('../kfolds/{}__{}.csv'.format(MODEL_NAME, str(round(np.mean(best_scores),5))), index=False)

print("Saving code to reproduce")
shutil.copyfile(os.path.basename(__file__),'../model_source/{}__{}.py'.format(MODEL_NAME, str(round(np.mean(best_scores),5))))

print("Saving submission file")
y_preds['target'] = y_preds.iloc[:, 1:].apply(np.mean, axis=1)
y_preds.rename(columns={'id':'ID_code'}, inplace=True)
y_preds[['ID_code', 'target']].to_csv('../model_predictions/submission_{}__{}.csv'.format(MODEL_NAME,str(round(np.mean(best_scores),5))), index=False)