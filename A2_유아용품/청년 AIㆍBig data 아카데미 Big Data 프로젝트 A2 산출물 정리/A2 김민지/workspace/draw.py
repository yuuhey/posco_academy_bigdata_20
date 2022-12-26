import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

# Font
matplotlib.rc("font", family = "NanumGothic")
matplotlib.rc("axes", unicode_minus = False)

df_churn = pd.read_csv('C:\workspace\이탈율.csv', encoding = 'euc-kr')
df_churn = df_churn.drop(labels = range(12,20), axis = 0) # 2020년 데이터 제외

# test 2
fig, host = plt.subplots(figsize=(15,5))
month = [1,2,3,4,5,6,7,8,9,10,11,12]

par1 = host.twinx()
par2 = host.twinx()
par3 = host.twinx()
# par4 = host.twinx()

host.set_xlabel("2019년 Month")
host.set_title("2019년 월별 고객 추이")

# p1, = host.plot(month, df_churn['이탈율'], label='이탈율', color='b', linewidth = '3.0')
p1, = host.plot(month, df_churn['복귀고객수'], label='복귀고객', color='b', linewidth = '3.0')
p2, = par1.plot(month, df_churn['신규고객수'], label='신규고객', color='r', linewidth = '3.0')
p3, = par2.plot(month, df_churn['재구매수'], label='재구매', color='c', linewidth = '3.0')
# p4, = par3.plot(month, df_churn['복귀고객수'], label='복귀고객', color='m', linewidth = '3.0')

lns = [p1, p2, p3]
# lns = [p1, p2, p3, p4]
host.legend(handles = lns, loc = 'best')

# figure 꽉 채우기
fig.tight_layout()
# y축 한개 삭제
plt.gca().axes.yaxis.set_visible(False)
plt.xticks(month)

plt.savefig("고객정보.png")