#!/usr/bin/env python
# coding: utf-8

# ## Load libraries

# In[1]:


import numpy as np
import pandas as pd
import re
from scipy import stats
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import math
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet, Ridge, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor as rfr,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, SCORERS

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor


get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
from IPython.display import display
display.max_columns = None
display.max_rows = None


# ## Load Data

# In[2]:


train = pd.read_csv('../data/train_processed.csv')
test_raw = pd.read_csv('../data/test.csv')
train_raw=pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test_processed.csv')
yt = stats.boxcox(train['SalePrice'], lmbda = 0.3) # Boxcox of SalePrice
act_pred = train['SalePrice']
X = train.drop(columns='SalePrice')
X = X.loc[:, X.columns!='HouseStyle_2.5Fin'] #HouseStyle_2.5Fin in training.csv but not in test.csv


# ## Skewness (adjust Skewness of numeric columns)
#  - Adjust features with skewness greater than 0.75

# In[3]:


numeric=X.loc[:, X.columns != 'IsPool']
numeric = numeric.loc[:, numeric.columns !='IsGarage']
numerical = numeric.dtypes[:27].index.to_list()
skewed = X[numerical].apply(lambda x: x.skew()).sort_values()
skewdf = pd.DataFrame({'Skew': skewed})
skewdf = skewdf[(skewdf)>0.75]
from scipy.special import boxcox1p
skewed = skewdf.index
lam = 0.15
for feat in skewed:
    X[feat] = boxcox1p(X[feat], lam)
    test[feat] = boxcox1p(test[feat], lam)
# newskewed = X[numerical].apply(lambda x: x.skew()).sort_values()


# ## Evaluate Models

# In[4]:


### This function returns Root Mean Squared Error for 5 fold cross validation tests.
### To compare side by side with Kaggle score.
def cv_rmse(model, X=X, yt=yt):
    y=train['SalePrice']
    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    RMSE_list=[]
    for train_id, holdout_id in kfold.split(X, yt):
        instance = clone(model)
        instance.fit(X.iloc[train_id], yt[train_id])
        y_pred = instance.predict(X.iloc[holdout_id])
        y_pred = inv_boxcox(y_pred, 0.3)
        y_mean=np.mean(y[holdout_id])
        RSE=np.sum((np.log(y_pred)-np.log(y[holdout_id]))**2)
        MSE=RSE/len(holdout_id)
        RMSE=np.sqrt(MSE)
        RMSE_list.append(RMSE)
    return RMSE_list


# ## Train Test Split

# In[5]:


x_tr, x_val, y_tr, y_val = train_test_split(X, yt, test_size=0.2, random_state=1)


# ## Model 1: Ridge 

# In[6]:


ridge = Ridge(normalize=True, alpha=0.049090909090909095)


# In[7]:


print(f"{np.mean(cv_rmse(ridge))} & {np.std(cv_rmse(ridge))}")


# ## Model 2: Lasso 

# In[8]:


lasso = Lasso(normalize=True, alpha=0.002454545454545455)


# In[9]:


np.mean(cv_rmse(lasso))


# ## Model 3: Elastic Net
# (This is essentially the same as Lasso)

# In[10]:


net = ElasticNet(alpha=0.002454545454545455, l1_ratio=1.0)


# In[11]:


np.mean(cv_rmse(net))


# ## Model 4: CatBoost

# In[12]:


catB = CatBoostRegressor(iterations=3000)


# In[13]:


np.mean(cv_rmse(catB))


# ## Model 5: Gradient Boosting

# In[14]:


gb = GradientBoostingRegressor(n_estimators = 4650, learning_rate = 0.04,
                                   max_depth = 2, max_features = 'sqrt',
                                   min_samples_leaf = 4, min_samples_split = 34, 
                                   loss = 'huber', random_state = 1)


# In[15]:


np.mean(cv_rmse(gb))


# ## Model 5: XGBoost

# In[16]:


xgboost = xgb.XGBRegressor(
    learning_rate =0.0492,
    n_estimators=2000,
    max_depth=2,
    min_child_weight=4,
    gamma=1.5789473684210527,
    subsample=0.8125,
    colsample_bytree=0.35486333333333336,
    random_state = 1,
    objective = "reg:squarederror")


# In[17]:


np.mean(cv_rmse(xgboost))


# ## LightGBM

# In[21]:


lightGBM = lgb.LGBMRegressor(objective = 'regression',
                         num_leaves = 5,
                         learning_rate = 0.05,
                         n_estimators = 1620,
                         max_bin = 55,
                         bagging_fraction = 0.889,
                         bagging_freq = 7,
                         feature_fraction = 0.23157894736842105,
                         feature_fraction_seed = 9,
                         bagging_seed = 9,
                         random_state = 1,
                         min_data_in_leaf = 7,
                         min_sum_hessian_in_leaf = 7)


# In[22]:


np.mean(cv_rmse(lightGBM))


# # Model Emsembling

# ## Simple Averaging Model

# In[23]:


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   


# In[24]:


simple_stack_model = AveragingModels(models = (lasso, net, catB, gb, xgboost, lightGBM))
np.mean(cv_rmse(simple_stack_model))


# In[25]:


simple_stack_model.fit(x_tr,y_tr)
prediction=simple_stack_model.predict(test)
prediction=inv_boxcox(prediction, 0.3)
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub.to_csv('../Submissions/submission_simpleStacked.csv',index=False)
prediction


# ## Stacked (meta_model=lasso)

# StackingModels Class Source: https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard

# In[26]:


class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original base models
    # We need the clone because we do not want to 
    # continuously fit and overwrite the original base models.
    def fit(self, X, yt):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=1)
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model.
        outoffold_pred = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_id, holdout_id in kfold.split(X, yt):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X.iloc[train_id], yt[train_id])
                y_pred = instance.predict(X.iloc[holdout_id])
                outoffold_pred[holdout_id, i] = y_pred  
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(outoffold_pred, yt)
        print(self.meta_model_.coef_)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)


# In[27]:


stacked_models = StackingAveragedModels(base_models = (lasso, net, gb, catB, xgboost),
                                                 meta_model = Lasso())
score = cv_rmse(stacked_models)
print(f"Stacked model RMSE Score: {np.mean(score)}")
# lasso, net, gb, xgboost, lightGBM - 0.10995646549877361
# ridge, lasso, net, gb, lightGBM - 0.1098998248584195
# stacked 1: ridge, lasso, net, gb, catB, xgboost - 0.10959475310991433
# stacked 2: lasso, net, gb, catB, xgboost - 0.10958984127519941


# In[28]:


stacked_models.fit(X, yt)
stacked_train_pred = stacked_models.predict(X)
stacked_pred = stacked_models.predict(test)
stacked_pred=inv_boxcox(stacked_pred, 0.3)
pred = pd.DataFrame(stacked_pred, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/pred_stacked_2.csv',index=False)
stacked_pred


# In[29]:


lightgbm=pd.read_csv('../Submissions/pred_lightgbm.csv')
stacked2=pd.read_csv('../Submissions/pred_stacked_2.csv')
prediction=0.7*stacked2.SalePrice+0.3*lightgbm.SalePrice
pred = pd.DataFrame(prediction, columns=['SalePrice'])
ID = test_raw[['Id']]
sub1=pd.merge(ID, pred, left_on = ID.index, right_on = pred.index).drop(columns=['key_0'])
sub1.to_csv('../Submissions/pred_final_prediction1.csv',index=False)


# In[ ]:




