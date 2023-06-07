#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import shap
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[2]:


project_dir = '/lustre/groups/epigenereg01/workspace/projects/vale/pausing/'


# In[3]:


output_dir = project_dir + 'linear_regression'

os.makedirs(output_dir, exist_ok=True)


# In[4]:


hepg2_df = pd.read_csv(project_dir + 'data/individual.HepG2.all.features.csv', index_col=0)


# In[5]:


k562_df = pd.read_csv(project_dir + 'data/individual.K562.all.features.csv', index_col=0)


# In[6]:


def clean_data(df):

    df = df.fillna(df.median())

    df.loc[np.isfinite(df['tx.ex.ratio'])==False,'tx.ex.ratio'] = 1
        
    return df


# In[7]:


def normalize_features(X_train, X_test):

    zero_variance_columns = X_train.columns[(X_train.std()==0).values]
    
    X_train = X_train.drop(columns=zero_variance_columns)
    X_test = X_test.drop(columns=zero_variance_columns)

    mean = X_train.mean()
    std = X_train.std()
    
    X_train_norm = (X_train-mean)/std
    
    X_test_norm = (X_test-mean)/std
    
    return X_train_norm, X_test_norm


# In[8]:


def grid_search_cv_alpha(X_train, y_train):
    
    model = Ridge(random_state=1)

    parameters = {'alpha':np.arange(1,10000,100)}

    clf = GridSearchCV(model, parameters, verbose=0)

    clf.fit(X_train, y_train)
    
    return clf


# In[42]:


def save_model(model, output_name):
    
    output_path = output_dir + '/' + output_name + '.pickle'
    
    with open(output_path, 'wb') as f:
        pickle.dump(model,f)
        
    print('Model saved to: ', output_path)


# In[44]:


def compute_shap_values(model, X_test, output_name):
    
    print('Computing Shapley values...')
    
    X100 = shap.utils.sample(X_test, 100)

    explainer = shap.Explainer(model.predict, X100, max_evals = 2 * X100.shape[1] + 1)
    
    shap_values = explainer(X_test, silent=True)
    
    #shap_values = (model.coef_ * X_test) - (model.coef_ * X_test).mean()
    
    shap_values = pd.DataFrame(shap_values.values, index=X_test.index, columns=X_test.columns)

    ##proteins = [x.split('.')[1] if 'chip' in x or 'clip' in x else 'NA' for x in X_test.columns]

    ##shap_df = pd.DataFrame({'protein':proteins, 'shap_values_sum':np.sum(shap_values.values,axis=0)})

    ##shap_df.groupby('protein').sum().to_csv(output_dir + '/' + output_name + '.csv')
    
    output_path = output_dir + '/' + output_name + '.csv.gz'
    
    shap_values.to_csv(output_path)
    
    print('Shapley values saved to: ', output_path)


# In[10]:


data = {'HepG2':clean_data(hepg2_df), 'K562':clean_data(k562_df)}

target_column = 'target_traveling.ratio'


# In[11]:


best_params = {'HepG2':{'alpha':2800},'K562':{'alpha':2800}}


# In[45]:


for cell_line in ('HepG2', 'K562'):

    print('First evaluation, 50/50 train/test split:', cell_line)
    
    df = data[cell_line]
    
    y = df[target_column]
    
    X = df.drop(columns=target_column)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=1)
    
    X_train, X_test = normalize_features(X_train, X_test)
    
    if len(best_params[cell_line])==0:
    
        print('Looking for optimal hyperparameters...')
    
        model = grid_search_cv_alpha(X_train, y_train)
        
        best_params[cell_line] = clf.best_params_
        
    else:
        
        model = Ridge(**best_params[cell_line],random_state=1)
        
        model.fit(X_train, y_train)
    
    print('Using best hyperparameters: ', best_params[cell_line])
    
    save_model(model,'train_on_half_' + cell_line + '_model' )
    
    train_score = model.score(X_train, y_train)
    
    test_score = model.score(X_test, y_test)
    
    print('Train score: ', round(train_score,3))

    print('Test score: ', round(test_score,3))
    
    compute_shap_values(model, X_test, 'train_on_' + cell_line + '_test_on_' + cell_line + '_shapley_values')
    
    print()


# In[48]:


common_features = set(data['HepG2'].columns).intersection(set(data['K562'].columns))
common_features.remove(target_column)


# In[49]:


common_genes = set(data['HepG2'].index).intersection(set(data['K562'].index))

unique_genes = {'HepG2':set(data['HepG2'].index).difference(set(data['K562'].index)),
               'K562':set(data['K562'].index).difference(set(data['HepG2'].index))}


# In[50]:


best_params_common = {'HepG2':{'alpha':2800},'K562':{'alpha':2800}}


# In[13]:


for train_cell_line, test_cell_line in (('HepG2','K562'), ('K562','HepG2')):

    print('Second evaluation, train on ', train_cell_line,' evaluate on ', test_cell_line )
    
    train_df = data[train_cell_line]
    
    y_train = train_df[target_column]
    X_train = train_df[common_features]
    
    test_df = data[test_cell_line]
    
    y_test = test_df[target_column]
    X_test = test_df[common_features]
    
    X_train, X_test = normalize_features(X_train, X_test)
    
    if len(best_params_common[train_cell_line])==0:
    
        print('Looking for optimal hyperparameters...')
    
        model = grid_search_cv_alpha(X_train, y_train)
        
        best_params_common[train_cell_line] = clf.best_params_
        
    else:
        
        model = Ridge(**best_params_common[train_cell_line],random_state=1)
        
        model.fit(X_train, y_train)
    
    print('Using best hyperparameters: ', best_params_common[train_cell_line])
    
    save_model(model,'train_on_full_' + train_cell_line + '_model' )
    
    train_score = model.score(X_train, y_train)
    
    test_score = model.score(X_test, y_test)
    
    print('Train score: ', round(train_score,3))

    print('Test score: ', round(test_score,3))
    
    test_score_common = model.score(X_test.loc[X_test.index.isin(common_genes)], y_test.loc[y_test.index.isin(common_genes)])

    print('Test score on common genes: ', round(test_score_common,3))
    
    test_score_unique = model.score(X_test.loc[X_test.index.isin(unique_genes[test_cell_line])], y_test.loc[y_test.index.isin(unique_genes[test_cell_line])])

    print('Test score on genes unique for the test cell line', '('+test_cell_line+'):', round(test_score_unique,3))
    
    print()

