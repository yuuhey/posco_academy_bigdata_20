import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Font
matplotlib.rc("font", family = "NanumGothic")
matplotlib.rc("axes", unicode_minus = False)

df_RFM = pd.read_csv('C:/workspace/Member_data4.csv', encoding = 'euc-kr')
df = pd.read_csv('C:/workspace/이탈율.csv', encoding = 'euc-kr')
df.head()
df.신규고객수.sort_values(ascending = False)
df.신규고객수.nunique()
plt.plot(df.신규고객수)
plt.show()

/81055




df_RFM.head()
df_RFM.columns

df_RFM.유입경로.value_counts()

df_per = []
df_per = df_RFM.유입경로.value_counts() / len(df_RFM.유입경로) * 100
df_per.round(2)*100
df_per_drop = df_RFM.유입경로['오픈마켓', '페이스북', '직접검색', '검색광고', '매장쿠폰', '이마트']


# df_RFM.총구매금액.sum()
df_RFM.groupby('총구매금액').sum()

100-(38.5+10.2+18)