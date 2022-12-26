import pandas as pd
import numpy as np
from datetime import datetime

# 고객별 구매일 모아서 다시만들기
data = '/home/piai/workspace/bigdata/Project/data/new_sales3.csv'
df_pro = pd.read_csv('/home/piai/workspace/bigdata/Project/data/new_Product2.csv')
df_member = pd.read_csv('/home/piai/workspace/bigdata/Project/data/Member_data2.csv')
df_sales = pd.read_csv(data, low_memory=False, encoding='euc-kr')

# 월별 전체 매출
df2 = df_sales.copy()
df2['구매일'] = df_sales['구매일'].astype(str).apply(lambda x:x[:7])
df2.groupby('구매일')['결제금액'].agg(['sum','count'])


# 월별 신규방문자, 신규구매, 재구매
"""
신규구매, 재 구매의 경우 특정 간격(90 days)에 따라 달라져야함
ex) 90일 지나면 기존 구매 고객도 신규 고객 취급
"""

df_member['가입월']=df_member['고객번호'].apply(lambda x:str(x)[:6])
df_member = df_member[df_member['가입월'] != '201812']
df_member['가입월'].value_counts()

df_new = pd.DataFrame()
df_new['전체결제수'] = df_sales['구매월'].value_counts()
df_new = df_new.sort_index()
df_new = df_new.reset_index()
df_new = df_new.rename(columns = {'index':'월'})

## 최초 구매
### Frequency
df_frequency = pd.DataFrame(df_sales.groupby(['고객번호'])['구매일'].apply(list)).reset_index()
df_frequency['가입일'] = df_frequency['고객번호'].apply(lambda x:str(x)[:8])
df_frequency['구매금액'] = df_sales.groupby('고객번호')['구매금액'].apply(list).values
### Convert to Datetime
df_frequency['구매일'] = df_frequency['구매일'].apply(lambda x:list(map(lambda y:np.datetime64(y, 'D'), x)))

## 월별 최초구매(간격계산 후) 고객, 해당 고객이 구매한 매출
df_sales['구매월'] = df_sales['구매일'].apply(lambda x:str(x)[:7].replace('-',''))
df_sales['할인율'] = 1-(df_sales['결제금액']/df_sales['구매금액']).round(2)

def first_sales(month, drop_days=90, df_frequency=df_frequency, df_sales = df_sales):
    # 최초구매일 in 특정월
    # - 특정일 사이에 있는지
    tmp = df_sales[df_sales['구매월'] == month].groupby('고객번호')['구매일'].min().reset_index()
    tmp_cash = df_sales[df_sales['구매월'] == month].groupby('고객번호')[['구매금액', '결제금액']].sum().reset_index()
    tmp['최대기간'] = tmp['구매일'].apply(lambda x:str(np.datetime64(x, 'D')-drop_days))
    tmp = pd.merge(tmp, tmp_cash, on='고객번호')

    def customer_state(x, df_frequency=df_frequency):
        date_list = df_frequency[df_frequency['고객번호'] == x['고객번호']]['구매일'].values[0]
        # 신규, 복귀, 재구매
        if sum(date_list < np.datetime64(x['구매일'], 'D')) > 0: # 구매일 전에 구매 기록있음
            if sum((np.datetime64(x['최대기간'], 'D') <= date_list) & (np.datetime64(x['구매일'], 'D') > date_list)) > 0: # 그 구매가 3개월 이내임
                return (False, True) # 신규아님, 재구매
            else:
                return (False, False) # 신규아님, 재구매 X == 복귀
        else:
            return (True, False) # 신규, 재구매 X
    
    tmp2 = tmp.apply(lambda x:customer_state(x), axis=1)
    tmp = tmp.drop(['구매일', '최대기간'], axis=1)
    tmp['신규여부'] = tmp2.apply(lambda x:x[0])
    tmp['재구매여부'] = tmp2.apply(lambda x:x[1])

    return tmp

df_new['월'] = df_new['월'].astype(str)
customer_list = [first_sales(month=i, drop_days=90) for i in df_new['월']]

df_new['이용고객수'] = [len(i) for i in customer_list]
df_new['신규고객수'] = [sum(i['신규여부']) for i in customer_list]
df_new['재구매고객수'] = [sum(i['재구매여부']) for i in customer_list]
df_new['복귀고객수'] = [(i[['신규여부', '재구매여부']].sum(axis=1) == 0).sum() for i in customer_list]
## 복귀 고객의 금액은 빠져있음
df_new['신규_구매금액'] = [i[i.apply(lambda x:x['신규여부'] == True, axis=1)]['구매금액'].sum() for i in customer_list]
df_new['재_구매금액'] = [i[i.apply(lambda x:x['재구매여부'] == True, axis=1)]['구매금액'].sum() for i in customer_list]
df_new['재구매_평균금액'] = df_new['재_구매금액']/df_new['재구매고객수']
df_new['신규_평균금액'] = df_new['신규_구매금액']/df_new['신규고객수']

ta = [1-(df_new['재구매고객수'][i]/df_new['이용고객수'][i-3]) for i in range(3, len(df_new))]
for _ in range(3):
    ta.insert(0, 0)
df_new['이탈율'] = ta


df_new.to_csv('./이탈율.csv', encoding='euc-kr', index=False)
# 추가로, customer_list 의 고객 ID를 활용해서
# 기존고객 , 신규고객 ID 구분.
# > 기존고객 구매액수, 신규고객 구매액수 구분




#=========================
df2 = df_sales.groupby('고객번호')['구매일'].apply(list)
df2.values[0]

sales_min = df_sales.groupby('고객번호')['구매일'].min().reset_index()

df_sales.groupby('구매월')['결제금액'].agg(['sum', 'count'])

df3 = df_sales.groupby('고객번호')['구매일'].min().apply(lambda x:str(x)[:7].replace('-',''))
df_new['최초구매월'] = df3.value_counts()

## 재구매 방문자 수
df_sales['구매일'] = df_sales['구매일'].astype('datetime64[D]')
g_sales = df_sales.groupby('고객번호')
### 2회이상 구매한 경우 Sales
df_sales2 = df_sales[df_sales['고객번호'].isin(g_sales['구매일'].count()[(g_sales['구매일'].count() > 1)].index)]

df_sales2























data = '/home/piai/workspace/bigdata/Project/data/rfm.csv'
df = pd.read_csv(data, encoding='euc-kr', low_memory=False, index_col=0)
df['가입일'] = df['고객번호'].apply(lambda x:x[:8])

df = df.rename(columns = {'마지막 구매일' : '최근구매'})
df = df.rename(columns = {'구매 구간' : '구매간격'})
df['최근구매'] = pd.to_datetime(df['최근구매'])

df['최근구매'] = df['최근구매'].apply(lambda x:np.datetime64(str(x).split(' ')[0], 'D'))
#df['마지막구매'] = df['마지막구매'].apply(lambda x:x.strftime('%Y-%m-%d'))

last_day = np.datetime64('2020-08-07', 'D')
df['최근구매'] = df['최근구매'].apply(lambda x:last_day-x)/np.timedelta64(1, 'D')
df['구매간격'] = df['구매간격']/np.timedelta64(1, 'D')

df.columns