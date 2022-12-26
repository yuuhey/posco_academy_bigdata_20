"""
기존 받은 세 데이터(Member, Product, Sales) 수정

1. 파생변수 추가 (구매일, 배송기간, 출고기간)
2. 이상치 제거(상품명이 중복된 경우 합치기)
3. Delivery.csv : 배송관련 모든 데이터를 담은 csv 파일 생성
4. 

"""

import pandas as pd
import numpy as np
from datetime import datetime


# Member_frequency.csv
df_member = pd.read_csv('/home/piai/workspace/bigdata/Project/data/Member_data2.csv', low_memory=False)
data = '/home/piai/workspace/bigdata/Project/data/new_sales3.csv'
df_sales = pd.read_csv(data, low_memory=False, encoding='euc-kr')
df_sales.columns
df_sales['구매일'] = df_sales['구매일'].astype('datetime64[D]')
df_new = pd.DataFrame(df_sales.groupby(['고객번호'])['구매일'].apply(list)).reset_index()
df_new['가입일'] = df_new['고객번호'].apply(lambda x:str(x)[:8])
df_new['구매금액'] = df_sales.groupby('고객번호')['구매금액'].apply(list).values
df_new.to_csv('/home/piai/workspace/bigdata/Project/data/member_frequency2.csv')

# Delivery.csv


df = pd.merge(df_member[['고객번호', '거주지역', '결제등록카드']], df_sales, on='고객번호', how='right')
df['배송시작일'] = df['배송시작일'].apply(lambda x:np.datetime64(x, 'D'))
df['배송완료일'] = df['배송완료일'].apply(lambda x:np.datetime64(x, 'D'))
df['구매일'] = df['구매일'].apply(lambda x:np.datetime64(x, 'D'))
df['배송기간'] = (df['배송완료일'] - df['배송시작일']).dt.days
df['출고기간'] = (df['배송시작일'] - df['구매일']).dt.days
df['구매요일'] = df['구매일'].apply(lambda x:datetime.datetime.weekday(x))
df.to_csv('/home/piai/workspace/bigdata/Project/data/delivery.csv', index=False, encoding='euc-kr')

df.groupby('거주지역')['배송기간'].mean()
df.groupby('거주지역')['배송기간'].count()
df.groupby('결제등록카드')['구매금액'].agg(['count', 'mean'])



# Product.csv
df2 = pd.read_csv('/home/piai/workspace/bigdata/Project/data/Product_data.csv', 
                 low_memory=False)

# 겹치는 상품명
np1 = np.unique(df2['상품명'].values, return_counts=True)
df2[df2['상품명'].apply(lambda x:x in np1[0][np1[1] >= 2])].to_csv('/home/piai/workspace/bigdata/Project/data/겹치는물품명.csv')
df3 = df2[df2['상품명'].apply(lambda x:x in np1[0][np1[1] >= 2])]

tmp = np1[0][np1[1] >=2]
tmp2 = [df3[df3['상품명'] == i]['물품판매량'].idxmax() for i in tmp]
sale = df3.groupby('상품명')['물품판매량'].sum()

df4 = df3[df3.index.isin(tmp2)]
df4['물품판매량'] = sale.values

# Remove All of Duplicate
df2 = df2[~df2.index.isin(df3.index)]
pd.concat([df2, df4], axis=0).to_csv('/home/piai/workspace/bigdata/Project/data/New_product.csv')


# New Sales.csv (3)
df = pd.read_csv('/home/piai/workspace/bigdata/Project/data/new_sales3.csv', encoding='euc-kr',
                 low_memory=False)
###df = df.rename(columns={"구매시월령(수정)":'after_born'})
df.columns

data2 = '/home/piai/workspace/bigdata/Project/data/new_product6.csv'
df2 = pd.read_csv(data2, encoding='euc-kr')

def check_product(df, df2):
    try:
        return(df2[df2['상품명'] == df]['물품대분류'].values[0])
    except:
        df['물품대분류']

new_col = []
for i in range(len(df)+1):
    new_col.append(check_product(df = df['상품명'].iloc[i], df2=df2))

df['물품대분류'] = pd.DataFrame({'물품대분류' : new_col})
df['구매월'] = df['구매일'].apply(lambda x:str(x)[:7].replace('-',''))
df['할인율'] = 1-(df['결제금액']/df['구매금액']).round(2)
df.to_csv('/home/piai/workspace/bigdata/Project/data/new_sales4.csv', index=False, encoding='euc-kr')
