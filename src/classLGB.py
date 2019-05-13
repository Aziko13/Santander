import lightgbm as lgb
from sklearn import model_selection
import pandas as pd
import gc
import numpy as np



class classLGB():

    def __init__(self, train_df, test_df, target_col, predictor_cols, id_col=None):
        self.train_df = train_df
        self.test_df = test_df
        self.target_col = target_col
        self.predictor_cols = predictor_cols

        if id_col is None:
            self.id_col = list(range(test_df.shape[0]))
        else:
            self.id_col = test_df[id_col]

    def kfold_train(self, nfolds, params, random_state=12, shuffle=True, nrounds=1000, early_stopping=100, oversampling=None):


        kf = model_selection.KFold(n_splits=nfolds, shuffle=shuffle, random_state=random_state)


        train_X = self.train_df[self.predictor_cols]
        train_y = self.train_df[self.target_col].values

        test_X = self.test_df[self.predictor_cols]

        best_iters = []
        best_scores = []

        y_preds = pd.DataFrame(columns=['id'])
        y_preds['id'] = self.id_col

        oof_preds = np.zeros(self.train_df.shape[0])
        feature_importance_df = pd.DataFrame()



        for i, (dev_index, val_index) in enumerate(kf.split(train_X)):
            dev_X, val_X = train_X.loc[dev_index, :], train_X.loc[val_index, :]
            dev_y, val_y = train_y[dev_index], train_y[val_index]

            lgtrain = lgb.Dataset(dev_X, label=dev_y)
            lgval = lgb.Dataset(val_X, label=val_y)
            evals_result = {}

            model = lgb.train(params, lgtrain, nrounds, valid_sets=[lgval],
                              early_stopping_rounds=early_stopping, verbose_eval=-1, evals_result=evals_result)


            oof_preds[val_index] = model.predict(val_X, num_iteration=model.best_iteration)
            pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = self.predictor_cols
            fold_importance_df["importance"] = np.log1p(model.feature_importance(importance_type='gain', iteration=model.best_iteration))
            fold_importance_df["fold"] = i + 1

            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

            best_iters.append(model.best_iteration)
            best_scores.append(model.best_score['valid_0'][params['metric']])

            y_preds['cv_' + str(i)] = pred_test_y

            gc.collect()

        return y_preds, model, best_iters, best_scores, feature_importance_df, oof_preds


    def params_tuning(self, nfolds=3):

        file = open('lgb_params_tuning.txt', 'a')
        file.write('New Params Tuning')
        file.write('\n')
        file.close()

        results = pd.DataFrame(columns=['params', 'score'])

        params_space = {'num_leaves': [5, 7, 40]
                        , 'min_data_in_leaf': [10, 50, 100, 150, 200, 300, 400, 500, 800, 1000, 1300]
                        , 'max_depth': [10, 13, 17, 19, 23, 30]
                        }
        # params_space = {'num_leaves': [5]
        #                 , 'min_data_in_leaf': [10, 50]
        #                 , 'max_depth': [10]
        #                             }

        params = {
            'learning_rate': 0.03,
            'objective': 'binary',
            'metric': 'auc',

            'boosting_type': 'gbdt',

            'min_data_in_leaf': 1300,
            'num_leaves': 5,
            'max_depth': 10,

            'feature_fraction': 0.3,
            'bagging_fraction': 0.3,
            'bagging_freq': 0
        }



        for num_leaves in params_space['num_leaves']:
            for min_data_in_leaf in params_space['min_data_in_leaf']:
                for max_depth in params_space['max_depth']:

                    params['num_leaves'] = num_leaves
                    params['min_data_in_leaf'] = min_data_in_leaf
                    params['max_depth'] = max_depth

                    y_preds, model, best_iters, best_scores, feature_importance_df, oof_preds = self.kfold_train(nfolds, params, random_state=12, shuffle=True, nrounds=10000, early_stopping=100)

                    score = np.mean(best_scores)

                    file = open('lgb_params_tuning.txt', 'a')
                    file.write(str(params))
                    file.write('\t')
                    file.write(str(score))
                    file.write('\t')
                    file.write('\n')
                    file.close()

                    results.loc[results.shape[0]] = [params, score]

        return results