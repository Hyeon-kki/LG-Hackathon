import pandas as pd 
import numpy as np
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import shap
# import eli5
# from eli5.sklearn import PermutationImportance
import warnings
from feature_selection_functions import rfe

data = pd.read_csv('train.csv')
features = ['bant_submit', 'business_unit', 'enterprise', 'lead_desc_length', 'customer_position', 'response_corporate', 'ver_cus', 'ver_pro']
dummy_features = ['enterprise', 'customer_position','response_corporate']
train_data = data[features]
train_data.rename(columns={'bant_submit': 'BantSubmit',
                       'business_unit':'BusinessUnit', 
                       'lead_desc_length':'LeadDescLength', 
                       'customer_position':'CustomerPosition', 
                       'response_corporate':'ResponseCorporate', 
                       'ver_cus':'VerCus', 
                       'ver_pro':'VerPro'}, inplace=True)

# 특수 문자를 대체할 문자열
replacement = {'/': '', ' ': '', '-':''}
# replace() 함수를 사용하여 특수 문자 대체
train_data.replace(replacement, regex=True, inplace=True)
# 원핫인코딩
train_data = pd.get_dummies(train_data, prefix_sep='', dtype=float)
columns_name = []
# 칼럼명에 특수문자 제거
for i in range(176):
    columns_name.append('columns'+str(i))
train_data.columns = columns_name

y_data = data['is_converted'].astype(int)
y_data.rename({'is_converted':'IsConverted'}, inplace=True)
x_train, x_val, y_train, y_val = train_test_split(train_data, y_data, random_state=0)

basic_archive = rfe(train_data, y_data, 'basic')