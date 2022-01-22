import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin, space_eval
from hyperopt.early_stop import no_progress_loss
from sklearn.metrics import mean_squared_error

def split_data(df, metadata_columns, production=False, target=['target']):
    
    metadata = df[metadata_columns]
    
    if production:
        X = df.drop(columns=metadata_columns)
        return X, metadata
    else:
        
        metadata_columns.extend(target)
        X = df.drop(columns=metadata_columns)
        y = df[target]
        return train_test_split(X, y, metadata, test_size=0.33, random_state=17)

def transform_data(X_test_raw, categorical_columns, X_train_raw=None, production=False):
    
    if production:
        with open('model/transformer.pkl', 'rb') as f:
            ct = pickle.load(f)
    
    else:    
        numerical_columns = [column for column in list(X_train_raw.columns) if column not in categorical_columns]

        ct = ColumnTransformer([
            ("scaler", StandardScaler(), numerical_columns),
            ("encoder", OneHotEncoder(drop='if_binary', sparse=False), categorical_columns)    
        ])

        with open('model/features_raw.pkl', 'wb') as f:
            pickle.dump(list(X_train_raw.columns), f)

        X_train_values = ct.fit_transform(X_train_raw)
        
        with open('model/transformer.pkl', 'wb') as f:
            pickle.dump(ct, f)
    
    X_test_values = ct.transform(X_test_raw)
    
    feature_names = ct.transformers_[0][2]
    encoded_features = ct.transformers_[1][1].get_feature_names()

    temp = []
    for index, feature in enumerate(ct.transformers_[1][2]):
        for column in encoded_features:
            if 'x'+str(index)+'_' in column:
                temp.append(column.replace('x'+str(index)+'_', feature+'_'))

    feature_names.extend(temp)
    
    if not production:
    
        X_train = pd.DataFrame(data=X_train_values, columns=feature_names)
        
        with open('model/features.pkl', 'wb') as f:
            pickle.dump(list(X_train.columns), f)
    
    X_test = pd.DataFrame(data=X_test_values, columns=feature_names)
    
    if production:
        return X_test
    else:   
        return X_train, X_test

def get_best_parameters(X_train, y_train, max_evals=128):                

    kf = KFold(n_splits=5, shuffle=True, random_state=17)
    
    space = {
        'n_estimators': hp.quniform('n_estimators', 100, 300, 5),
        'max_depth': hp.quniform('max_depth', 1, 20, 1),
        'learning_rate': hp.uniform('learning_rate', 0, 1),
        'booster': hp.choice('booster', ['gbtree']),
        'gamma': hp.uniform('gamma', 0, 0.50),
        'min_child_weight': hp.uniform('min_child_weight', 0, 10),
        'subsample': hp.uniform('subsample', 0, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0, 5),
        'reg_lambda': hp.uniform('reg_lambda', 0, 5),
    }

    def train_XGBoost(space):
    
        model = XGBRegressor(
            n_estimators      = int(space['n_estimators']),
            max_depth         = int(space['max_depth']),
            learning_rate     = space['learning_rate'],
            booster           = space['booster'],
            gamma             = space['gamma'],
            min_child_weight  = space['min_child_weight'],
            subsample         = space['subsample'],
            colsample_bytree  = space['colsample_bytree'],
            reg_alpha         = space['reg_alpha'],
            reg_lambda        = space['reg_lambda'],
            eval_metric       = 'rmse',
            random_state      = 17)

        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_root_mean_squared_error')

        return{
            'loss': -np.mean(scores), 
            'loss_variance': np.var(scores, ddof=1),
            'status': STATUS_OK}

    best_hyperparameters = fmin(
        fn=train_XGBoost,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        early_stop_fn=no_progress_loss(30)
        )
    
    best_params = space_eval(space, best_hyperparameters)
    
    with open('model/best_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)

    return best_params

def get_model(best_params, X_train, y_train):
    
    model = XGBRegressor(
        n_estimators      = int(best_params['n_estimators']),
        max_depth         = int(best_params['max_depth']),
        learning_rate     = best_params['learning_rate'],
        booster           = best_params['booster'],
        gamma             = best_params['gamma'],
        min_child_weight  = best_params['min_child_weight'],
        subsample         = best_params['subsample'],
        colsample_bytree  = best_params['colsample_bytree'],
        reg_alpha         = best_params['reg_alpha'],
        reg_lambda        = best_params['reg_lambda'],
        objective         = 'reg:squarederror',
        eval_metric       = 'rmse',
        random_state      = 17)

    model.fit(X_train, y_train)
    
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    print('RMSE: {}'.format(mean_squared_error(y_test, y_pred, squared=False)))
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, max_display=200, plot_type='bar', show=False, sort=True)
    plt.savefig('model/shap_values.png')
    
