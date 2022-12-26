import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import wilcoxon

matplotlib.rc('font', family='NanumGothic')
matplotlib.rc('axes', unicode_minus=False)

font = {'family' : 'NanumGothic',
        'size' : 14}
matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': 22})

df = pd.read_csv('/home/piai/workspace/bigdata/Project/data/이탈율.csv', encoding='euc-kr')

df['월'] = df['월'].astype(str)
df['월'] = df['월'].apply(lambda x:x[:4] + '-' + x[4:])
df['월'] = df['월'].apply(lambda x:np.datetime64(x, 'M'))

# len-1 : 2019-01
x_label = ['재구매자 결제금액', '신규가입자 결제금액']
x_data = [df['재구매_평균결제금액'].sum()/len(df)-1, df['신규_평균결제금액'].sum()/len(df)]
x_data = [df['재구매_평균결제금액'], df['신규_평균결제금액']]
bar_df = pd.DataFrame(x_data).T.iloc[1:]

## Barplot 재구매자, 신규구매자
plt.figure(figsize=(5,8))
plt.bar(height=bar_df.mean(), x=bar_df.columns,
        align='center', color=['#e35f62', 'y'], width=0.5, capsize=10,
        edgecolor='lightgray', linewidth=3, yerr=bar_df.std())
plt.xticks([])
#plt.savefig('./data/graph/재구매신규구매boxplot.png')
plt.show()

# 윌콕슨 Test
wilcoxon(bar_df.iloc[:,0], bar_df.iloc[:,1])
# P-value = 3.814*e^-06

bar_df.mean()
# Pieplot 고객 비율
tmp = sum(df['신규고객수']+df['복귀고객수']) / sum(df['이용고객수'])
plt.figure(figsize=(5,5))
plt.pie([tmp, 1-tmp], 
        labels=('신규고객', '기존고객'),
        explode=(0.1, 0.1),
        shadow=True,
        autopct='%1.2f%%'
        #startangle=90,
        #textprops={'fontsize':14}
        )
plt.show()
#plt.savefig('./test.png')

df.columns
df = df.iloc[:12]
## PLT line graph
plt.figure(figsize=(15,10))
ax = plt.plot(df['월'], df['이용고객수'], color='b', label = '전체고객수')
plt.plot(df['월'], df['이용고객수'], 'o', color='b')
plt.plot(df['월'], df['신규고객수'], 'o')
plt.plot(df['월'], df['신규고객수'], label = '신규고객수')
ax2 = plt.twinx()
ax2 = sns.lineplot(data=df, x='월', y='이탈율', label='이탈율', marker='o', linestyle='dashed', color='red')
ax2.set_ylim((0, 1))
plt.legend()
ax2.legend()
plt.show()