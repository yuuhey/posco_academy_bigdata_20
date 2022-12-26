import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Font
matplotlib.rc("font", family = "NanumGothic")
matplotlib.rc("axes", unicode_minus = False)


df_churn = pd.read_csv('C:\workspace\이탈율.csv', encoding = 'euc-kr')
df_churn.head(10)
# df_churn = df_churn.drop(labels = range(12,20), axis = 0) # 2020년 데이터 제외

month = ['19-01', '19-02', '19-03','19-04', '19-05', '19-06','19-07', '19-08', '19-09',
        '19-10', '19-11', '19-12','20-01', '20-02', '20-03', '20-04', '20-05', '20-06', '20-07', '20-08']

churn_per = df_churn.이탈율*100
churn_per = churn_per.round(2)
churn_per.loc[3:] # 2019년4월~2020년 8월

CHURN = churn_per.loc[3:].tolist()
CHURN

# 방법 3
fig = plt.figure(figsize = (20, 5))
ax = fig.add_subplot(1,1,1)
ax.plot(df_churn['이탈율'], label='이탈율', color='b')
ax.plot(df_churn['신규고객수'], label='신규고객', color='r')
ax.plot(df_churn['재구매수'], label='재구매', color='g')
ax.plot(df_churn['복귀고객수'], label='복귀고객', color='y')

ax.set_title('월별 고객 추이', fontsize = 20)
ax.set_xlabel('month', fontsize = 14)

ax.legend(fontsize=12, loc='best', option = 'auto')
ax.legend(fontsize = 12)

plt.yscale('log')
# plt.show()
plt.savefig('고객정보.png')