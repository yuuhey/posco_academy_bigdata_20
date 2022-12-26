import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Font
matplotlib.rc("font", family = "NanumGothic")
matplotlib.rc("axes", unicode_minus = False)

df_sales = pd.read_csv('C:/workspace/new_sales4.csv', encoding = 'euc-kr')
df_memebers = pd.read_csv('C:/workspace/Member_data3.csv', encoding = 'euc-kr')
df_product = pd.read_csv('C:/workspace/new_product6.csv', encoding = 'euc-kr')

