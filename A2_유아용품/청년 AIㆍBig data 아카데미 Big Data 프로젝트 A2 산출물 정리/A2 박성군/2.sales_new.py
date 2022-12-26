import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib
matplotlib.rc('font', family='NanumGothic')
matplotlib.rc('axes', unicode_minus=False)


# Product.csv
df2 = pd.read_csv('/home/piai/workspace/bigdata/Project/data/Product_data.csv', 
                 low_memory=False)

# 겹치는 상품명

np1 = np.unique(df2['상품명'].values, return_counts=True)
df2[df2['상품명'].apply(lambda x:x in np1[0][np1[1] >= 2])].to_csv('/home/piai/workspace/bigdata/Project/data/겹치는물품명.csv')
df3 = df2[df2['상품명'].apply(lambda x:x in np1[0][np1[1] >= 2])]

# duplicated

tmp = np1[0][np1[1] >=2]
tmp2 = [df3[df3['상품명'] == i]['물품판매량'].idxmax() for i in tmp]
sale = df3.groupby('상품명')['물품판매량'].sum()

df4 = df3[df3.index.isin(tmp2)]
df4['물품판매량'] = sale.values

# Remove All of Duplicate
df2 = df2[~df2.index.isin(df3.index)]
pd.concat([df2, df4], axis=0).to_csv('/home/piai/workspace/bigdata/Project/data/New_product.csv')

# New Sales.csv

df = pd.read_csv('/home/piai/workspace/bigdata/Project/data/Sales_data.csv', 
                 low_memory=False)
df = df.rename(columns={"구매시월령(수정)":'after_born'})
df.columns

df.groupby('상품명')['구매금액'].agg(['min', 'mean'])
df['추정수량'] = df.groupby('상품명')['구매금액'].apply(lambda x:x/min(x))

data2 = '/home/piai/workspace/bigdata/Project/data/Product_data_new_1112.csv'
df2 = pd.read_csv(data2)

def check_product(df, df2):
    if df == '맘큐 허그박스':
        return ('맘큐')
    try:
        return(df2[df2['상품명'] == df]['물품대분류'].values[0])
    except:
        return ('결측')

new_col = []
for i in range(len(df)):
    new_col.append(check_product(df = df['상품명'].iloc[i], df2=df2))

df['물품대분류'] = pd.DataFrame({'물품대분류' : new_col})
df.to_csv('/home/piai/workspace/bigdata/Project/data/new_sales.csv')

# New Product CSV
df_new = df.groupby('상품명')[['구매금액','추정수량']].apply(lambda x:sum(x['구매금액'])/sum(x['추정수량']))
df_new = df_new.fillna(0)
df_new = df_new[~(df_new.index == '맘큐 허그박스')]

df2['추정단가'] = df_new.values

df2.to_csv('/home/piai/workspace/bigdata/Project/data/new_product.csv')