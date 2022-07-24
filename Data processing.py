import os
import numpy as np
import pandas as pd
import gc
from datetime import datetime
pd.set_option('display.max_columns',None)
# load the data dictionary
train_dict=pd.read_excel('E:\Python_env\python_project\COMP 9417\Project\Data_Dictionary.xlsx',header=2,sheet_name='train')
history_dict=pd.read_excel('E:\Python_env\python_project\COMP 9417\Project\Data_Dictionary.xlsx',header=2,sheet_name='history')
new_merchant_dict=pd.read_excel('E:\Python_env\python_project\COMP 9417\Project\Data_Dictionary.xlsx',header=2,sheet_name='new_merchant_period')
merchant_dict=pd.read_excel('E:\Python_env\python_project\COMP 9417\Project\Data_Dictionary.xlsx',header=2,sheet_name='merchant')
#print(train_dict,'\n',history_dict,'\n',new_merchant_dict,'\n',merchant_dict)
# From the data we know that not all the features are numbers and some features are discrete.
# According to this information, we need to transform some features into numbers and classify the data by its type.
train=pd.read_csv('E:\Python_env\python_project\COMP 9417\Project/train.csv')
test=pd.read_csv('E:\Python_env\python_project\COMP 9417\Project/test.csv')
merchant = pd.read_csv('E:\Python_env\python_project\COMP 9417\Project\merchants.csv')
new_transaction = pd.read_csv('E:\Python_env\python_project\COMP 9417\Project/new_merchant_transactions.csv')
history_transaction = pd.read_csv('E:\Python_env\python_project\COMP 9417\Project/historical_transactions.csv')
def type_transform_dictionary_sort(feature):
    value = feature.unique().tolist()
    value.sort()
    return feature.map(pd.Series(range(len(value)), index=value)).values

##Processing 'train.csv','test.csv'

#Checking if card_id is unique.
print("If the feature 'card_id' is unique: ",test['card_id'].nunique()+train['card_id'].nunique() == len(test['card_id'].values.tolist()+train['card_id'].values.tolist()))
print("If the feature 'merchabt_id' is unique: ", merchant.shape[0] == merchant['merchant_id'].nunique())
#Whether there is a value but missing
print("Whether there is a value but missing in the data-set 'train':\n",train.isnull().sum())
print("Whether there is a value but missing in the data-set 'test':\n",test.isnull().sum())
# Noticing that data-set 'test' has one missing value in feature 'first_active_month'. But the amount is small,so we can ignore the missing data.
# Noticing that data-set 'test' has one missing value in feature 'first_active_month'. But the amount is small,so we can ignore the missing data.
# Transforming the feature 'first_active_month' into numeric and sort the value by dictionary order.
train['first_active_month']=type_transform_dictionary_sort(train['first_active_month'].astype(str))
test['first_active_month']=type_transform_dictionary_sort(test['first_active_month'].astype(str))
#export the processed data and clean the mermory.
train.to_csv("E:\Python_env\python_project\COMP 9417\Project/processed_train.csv", index=False)
test.to_csv("E:\Python_env\python_project\COMP 9417\Project/processed_test.csv", index=False)
del train
del test
gc.collect()

##Processing 'merchant.csv'
#Because there are many text data in the data-set 'merchant.csv', so we classify features by its own type
text_data = ['merchant_id', 'merchant_group_id', 'merchant_category_id',
       'subsector_id', 'category_1',
       'most_recent_sales_range', 'most_recent_purchases_range',
       'category_4', 'city_id', 'state_id', 'category_2']
numeric_data = ['numerical_1', 'numerical_2',
     'avg_sales_lag3', 'avg_purchases_lag3', 'active_months_lag3',
       'avg_sales_lag6', 'avg_purchases_lag6', 'active_months_lag6',
       'avg_sales_lag12', 'avg_purchases_lag12', 'active_months_lag12']
# Transforming the feature 'first_active_month' into numeric and sort the value by dictionary order.
features=['category_1', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
for column in features:
    merchant[column]=type_transform_dictionary_sort(merchant[column])
#Whether there is a value but missing
print("Whether there is a value but missing in the data-set 'merchant':\n",merchant.isnull().sum())
print("Checking abnormal numerical values of the data-set 'merchant':\n",merchant[numeric_data].describe())
# Fill in the missing value of non-numerical data
merchant['category_2'] = merchant['category_2'].fillna(-1)
#Using maxiium value to replace the infite value
infite_column=['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']
merchant[infite_column] = merchant[infite_column].replace(np.inf, merchant[infite_column].replace(np.inf, -99).max().max())
#In the data-set 'merchant', there is 13 missing vaules in features 'avg_sales_lag3','avg_sales_lag6','avg_sales_lag12'.
#Accodring to the data dictionary we know that these three features is about the average value about sales.
#There is a highly possiblity that these 13 missing vaules is from 13 new mercahnt.
#Using mean to replace the missing value
for column in numeric_data:
    merchant[column]=merchant[column].fillna(merchant[column].mean())
print("Checking abnormal numerical values of the data-set 'merchant':\n",merchant[numeric_data].describe())
#Each merchant_id should have only the unique information about the merchant information.
print("if each merchant_id has unique relevant information:",merchant.shape[0] == merchant['merchant_id'].nunique())
duplicate_column=['merchant_id', 'merchant_category_id', 'subsector_id', 'category_1', 'city_id', 'state_id', 'category_2']
merchant = merchant.drop(duplicate_column[1:], axis=1)
merchant = merchant.loc[merchant['merchant_id'].drop_duplicates().index.tolist()].reset_index(drop=True)

transaction=pd.concat([new_transaction,history_transaction],axis=0,ignore_index=True)
del new_merchant_dict
del history_transaction
gc.collect()
#Because there are many text data in the data-set 'new_transaction.csv','history_transaction.csv', so we classify features by its own type
numeric_data =  [ 'installments', 'month_lag', 'purchase_amount']
text_data = ['authorized_flag', 'card_id', 'city_id', 'category_1','category_3', 'merchant_category_id', 'merchant_id', 'category_2', 'state_id','subsector_id']
time_data=['purchase_date']
# Transforming the feature 'first_active_month' into numeric and sort the value by dictionary order.
features=['authorized_flag', 'category_1', 'category_3']
for column in features:
    transaction[column]=type_transform_dictionary_sort(transaction[column].fillna(-1).astype(str))
#full in the missing value
#transaction[features]=transaction[features].fillna(-1).astype(str)
transaction[text_data]=transaction[text_data].fillna(-1)
# processing the time-data
transaction['purchase_month'] = transaction['purchase_date'].apply(lambda x:'-'.join(x.split(' ')[0].split('-')[:2]))
transaction['purchase_hour_section'] = transaction['purchase_date'].apply(lambda x: x.split(' ')[1].split(':')[0]).astype(int)//6
transaction['purchase_day'] = transaction['purchase_date'].apply(lambda x: datetime.strptime(x.split(" ")[0], "%Y-%m-%d").weekday())//5
del transaction['purchase_date']
transaction['purchase_month'] = type_transform_dictionary_sort(transaction['purchase_month'].fillna(-1).astype(str))
#Merge 'merchant.csv' with 'transaction.csv'
cols = ['merchant_id', 'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
transaction = pd.merge(transaction, merchant[cols], how='left', on='merchant_id')

numeric_cols = ['purchase_amount', 'installments']

category_cols = ['authorized_flag', 'city_id', 'category_1',
       'category_3', 'merchant_category_id','month_lag','most_recent_sales_range',
                 'most_recent_purchases_range', 'category_4',
                 'purchase_month', 'purchase_hour_section', 'purchase_day']

id_cols = ['card_id', 'merchant_id']

transaction[cols[1:]] = transaction[cols[1:]].fillna(-1).astype(str)
transaction[category_cols] =transaction[category_cols].fillna(-1)
transaction.to_csv("E:\Python_env\python_project\COMP 9417\Project/processed_transaction.csv", index=False)
del transaction
gc.collect()