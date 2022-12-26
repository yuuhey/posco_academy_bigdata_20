import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.svm import SVC

# 결측치처리 + 인코딩 / 스케일링 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# 불균형 데이터 처리
from imblearn.over_sampling import SMOTE
from sklearn.compose import make_column_transformer


# 교차검증 + 하이퍼 파라미터튜닝 

df_mem = pd.read_csv('/home/piai/workspace/bigdata/Project/data/Member_data4.csv', encoding='euc-kr')
df_mem['첫구매월령'] = df_mem['첫구매월령'].fillna('NA')

def change_date(x):
    if x == 'NA':
        return ('NA')
    if x < 0:
        return ('baby0')
    for i in range(1, 26):
        if x <=i*3:
            return (f'baby{i}')

def set_churn(x):
    if x == 2:
        return (1)
    else:
        return (x)

df_mem['첫구매월령'] = df_mem['첫구매월령'].apply(lambda x:change_date(x))
df_mem2 = df_mem.copy()
df_mem2['chun'] = df_mem2['chun'].apply(lambda x:set_churn(x))

df_x_1 = df_mem2[['성별', '결혼유무', '거주지역', '유입경로', '자녀여부', '첫구매물품', '첫구매월령']]
df_x_numeric = df_mem2[['연령', '첫주문까지일', '첫결제금액']]
df_y = df_mem2['chun']

numeric_pipe = make_pipeline( SimpleImputer() , MinMaxScaler() )
category_pipe = make_pipeline( SimpleImputer(strategy='most_frequent') , OneHotEncoder())

numeric_list = df_x_numeric.describe().columns.tolist()
category_list = df_x_1.describe(include='object').columns.tolist()

preprocess_pipe = make_column_transformer( (numeric_pipe, numeric_list),
                                           (category_pipe, category_list))
model_pipe = make_pipeline(preprocess_pipe, SMOTE(), GradientBoostingClassifier())
df_train_x, df_test_x, df_train_y, df_test_y = train_test_split(pd.concat([df_x_numeric, df_x_1], axis=1), df_y, test_size = 0.4, random_state = 1234)

model_pipe.fit(df_train_x, df_train_y)
print ('Train Score', model_pipe.score(df_train_x, df_train_y))
print ('Test Score', model_pipe.score(df_test_x, df_test_y))
print ('F1 Score : ', f1_score(df_test_y, model_pipe.predict(df_test_x)))
print (confusion_matrix(df_test_y, model_pipe.predict(df_test_x)))


#param_grid or param_distribution
grid_model = RandomizedSearchCV(model_pipe ,scoring='f1', param_distributions={
    # 'mlpclassifier__hidden_layer_sizes':[(i, i) for i in range(5, 100, 15)], #5
    # 'mlpclassifier__max_iter': [1000],
    # 'mlpclassifier__solver': ['sgd', 'adam'], #3
    # "mlpclassifier__batch_size":[i for i in range(500, 2001, 500)], #5
    # 'mlpclassifier__activation':['logistic', 'tanh', 'relu'] #3
    # # 17500
    'gradientboostingclassifier__n_estimators':[60, 80, 100, 150, 250, 500], # 3
    'gradientboostingclassifier__max_depth' : [i for i in range(3, 20, 2)], # 4
    'gradientboostingclassifier__min_samples_leaf' : [i for i in range(5, 200, 10)], # 5
    'gradientboostingclassifier__min_samples_split' : [i for i in range(15, 500, 10)], # 10
    'gradientboostingclassifier__learning_rate' : [i*0.1 for i in range(1, 10)] # 5
    #Randomforest
    # 'randomforestclassifier__n_estimators': [80, 100, 150, 200],
    # 'randomforestclassifier__max_features': [i for i in range(1, 10, 3)],
    # 'randomforestclassifier__max_depth': [3, 5, 10, 15]
    #'randomforestclassifier__min_samples_leaf':,
    #'randomforestclassifier__min_sam':,
    #'randomforestclassifier__n_estimators':,
    # SVC
    # 'svc__gamma' :[0.05 * gamma for gamma in range(1,21)],
    # 'svc__C': [c*0.1 for c in range(1,21)]
    } , cv=3 , n_jobs=10)

grid_model.fit(df_train_x, df_train_y)
best_model = grid_model.best_estimator_
best_model.score(df_test_x, df_test_y)
grid_model.best_params_

print ('F1 score : ', f1_score(df_test_y, best_model.predict(df_test_x)))
print ('Precision : ', precision_score(df_test_y, best_model.predict(df_test_x)))
print ('Recall : ', recall_score(df_test_y, best_model.predict(df_test_x)))
print (confusion_matrix(df_test_y, best_model.predict(df_test_x)))

import pickle
with open('./data/churn_model2.pickle', mode='wb') as f:
    pickle.dump(grid_model, f)

df_mem['predict'] = pd.DataFrame(best_model.predict_proba(pd.concat([df_test_x, df_train_x])))[0]
df_mem.to_csv('./data/df_mem_predict.csv', encoding='euc-kr', index=False)


#=========================================
import pickle
import joblib

best_model = joblib.load('/home/piai/workspace/posco-service/bigdata/data/churn_model_final.pkl')
dir(best_model)
best_model.best_estimator_

tmp = best_model.predict_proba(pd.concat([df_train_x, df_test_x], axis=0).sort_index())
tmp = pd.DataFrame(tmp)

df_mem['churn_predict0'] = tmp[0]
df_mem['churn_predict1'] = tmp[1]

df_mem.to_csv('./data/churn_predict.csv',encoding='euc-kr', index=False)

df_mem[df_mem['chun'] == 2]['churn_predict1'].plot.box()
plt.savefig('./data/churn_predict2.png')
plt.show()
df_mem.plot.scatter(x='churn_predict0', y='chun')