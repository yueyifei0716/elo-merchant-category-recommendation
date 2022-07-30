#!/usr/bin/env python
# coding: utf-8

# In[32]:


import gc
import time
import numpy as np
import pandas as pd
from datetime import datetime

# In[33]:


train = pd.read_csv('preprocess/train_pre.csv')
test = pd.read_csv('preprocess/test_pre.csv')
transaction = pd.read_csv('preprocess/transaction_d_pre.csv')

# In[34]:


# 标注离散字段or连续型字段
numeric_cols = ['purchase_amount', 'installments']

category_cols = ['authorized_flag', 'city_id', 'category_1',
                 'category_3', 'merchant_category_id', 'month_lag', 'most_recent_sales_range',
                 'most_recent_purchases_range', 'category_4',
                 'purchase_month', 'purchase_hour_section', 'purchase_day']

id_cols = ['card_id', 'merchant_id']

# In[42]:


transaction.loc[1].values

# In[43]:


transaction.shape[0]

# In[44]:


# 创建字典用于保存数据
features = {}
card_all = train['card_id'].append(test['card_id']).values.tolist()
for card in card_all:
    features[card] = {}

# 标记不同类型字段的索引
columns = transaction.columns.tolist()
idx = columns.index('card_id')
category_cols_index = [columns.index(col) for col in category_cols]
numeric_cols_index = [columns.index(col) for col in numeric_cols]

# 记录运行时间
s = time.time()
num = 0

# 执行循环，并在此过程中记录时间
for i in range(transaction.shape[0]):
    va = transaction.loc[i].values
    card = va[idx]
    for cate_ind in category_cols_index:
        for num_ind in numeric_cols_index:
            # print(type(columns[cate_ind]), type(va[cate_ind]), type(columns[num_ind]))
            col_name = '&'.join([columns[cate_ind], str(va[cate_ind]), columns[num_ind]])
            features[card][col_name] = features[card].get(col_name, 0) + va[num_ind]
    num += 1
    if num % 1000000 == 0:
        print(time.time() - s, "s")
del transaction
gc.collect()

# In[45]:


# 字典转dataframe
df = pd.DataFrame(features).T.reset_index()
del features
cols = df.columns.tolist()
df.columns = ['card_id'] + cols[1:]

# 生成训练集与测试集
train = pd.merge(train, df, how='left', on='card_id')
test = pd.merge(test, df, how='left', on='card_id')
del df
train.to_csv("preprocess/train_dict.csv", index=False)
test.to_csv("preprocess/test_dict.csv", index=False)

gc.collect()

# In[46]:


transaction = pd.read_csv('preprocess/transaction_g_pre.csv')

# In[47]:


# 标注离散字段or连续型字段
numeric_cols = ['authorized_flag', 'category_1', 'installments',
                'category_3', 'month_lag', 'purchase_month', 'purchase_day', 'purchase_day_diff', 'purchase_month_diff',
                'purchase_amount', 'category_2',
                'purchase_month', 'purchase_hour_section', 'purchase_day',
                'most_recent_sales_range', 'most_recent_purchases_range', 'category_4']
categorical_cols = ['city_id', 'merchant_category_id', 'merchant_id', 'state_id', 'subsector_id']

# In[48]:


# 创建空字典
aggs = {}

# 连续/离散字段统计量提取范围
for col in numeric_cols:
    aggs[col] = ['nunique', 'mean', 'min', 'max', 'var', 'skew', 'sum']
for col in categorical_cols:
    aggs[col] = ['nunique']
aggs['card_id'] = ['size', 'count']
cols = ['card_id']

# 借助groupby实现统计量计算
for key in aggs.keys():
    cols.extend([key + '_' + stat for stat in aggs[key]])

df = transaction[transaction['month_lag'] < 0].groupby('card_id').agg(aggs).reset_index()
df.columns = cols[:1] + [co + '_hist' for co in cols[1:]]

df2 = transaction[transaction['month_lag'] >= 0].groupby('card_id').agg(aggs).reset_index()
df2.columns = cols[:1] + [co + '_new' for co in cols[1:]]
df = pd.merge(df, df2, how='left', on='card_id')

df2 = transaction.groupby('card_id').agg(aggs).reset_index()
df2.columns = cols
df = pd.merge(df, df2, how='left', on='card_id')
del transaction
gc.collect()

# 生成训练集与测试集
train = pd.merge(train, df, how='left', on='card_id')
test = pd.merge(test, df, how='left', on='card_id')
del df
train.to_csv("preprocess/train_groupby.csv", index=False)
test.to_csv("preprocess/test_groupby.csv", index=False)

gc.collect()

# In[49]:


train_dict = pd.read_csv("preprocess/train_dict.csv")
test_dict = pd.read_csv("preprocess/test_dict.csv")
train_groupby = pd.read_csv("preprocess/train_groupby.csv")
test_groupby = pd.read_csv("preprocess/test_groupby.csv")

# In[50]:


for co in train_dict.columns:
    if co in train_groupby.columns and co != 'card_id':
        del train_groupby[co]
for co in test_dict.columns:
    if co in test_groupby.columns and co != 'card_id':
        del test_groupby[co]

# In[51]:


train = pd.merge(train_dict, train_groupby, how='left', on='card_id').fillna(0)
test = pd.merge(test_dict, test_groupby, how='left', on='card_id').fillna(0)

# In[52]:


train.to_csv("preprocess/train.csv", index=False)
test.to_csv("preprocess/test.csv", index=False)

del train_dict, test_dict, train_groupby, test_groupby
gc.collect()

# In[ ]:
