import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,f1_score,precision_score
# class Wrangler:
#     def __init__(self, directory_path):
#         self.directory_path = directory_path
#         self.dfs = []

#     def test(self):
#         print(directory_path)


# if __name__ == '__main__':
#     directory_path = 'BigSupplyCo_Data_CSVs/'

#     test_class = Wrangler(directory_path)
#     test_class.test()

#     ## forktree 2022-03-11 11:19
#     print("Test")

df_order = pd.read_csv('BigSupplyCo_Data_CSVs/BigSupplyCo_Orders.csv')

# # # Explornatory Data Analysis
# =============================================================================

df_order.info()
df_order.head()
df_order.describe()

# check na
df_order.apply(lambda x: sum(x.isnull()))

# delivery status
sns.countplot(x='Delivery Status', data = df_order)

# delivery status vs type
sns.countplot(x='Type', hue='Delivery Status', data = df_order)

# delivery status vs market
sns.countplot(x='Market', hue='Delivery Status', data = df_order)

# delivery status vs country
sns.countplot(x='Order Country', hue='Delivery Status', data = df_order)

# delivery status vs order department
sns.countplot(x='Order Department Id', hue='Delivery Status', data = df_order)

# delivery status vs late delivery risk
sns.countplot(x='Late Delivery Risk', hue='Delivery Status', data = df_order)

# time series effect?
df_order['order_year']= pd.DatetimeIndex(df_order['order date (DateOrders)']).year
df_order['order_month'] = pd.DatetimeIndex(df_order['order date (DateOrders)']).month
df_order['order_week_day'] = pd.DatetimeIndex(df_order['order date (DateOrders)']).weekday
df_order['order_hour'] = pd.DatetimeIndex(df_order['order date (DateOrders)']).hour


sns.countplot(x='order_year', hue='Delivery Status', data = df_order)
sns.countplot(x='order_month', hue='Delivery Status', data = df_order)

df_time = df_order.value_counts(subset=['Delivery Status', 'order_month','order_year'], sort=False)
df_time = df_time.reset_index()
df_time.rename(columns={0:'count'})
sns.lineplot(data=df_time, x='order_month', y=0, hue = 'order_year', style='Delivery Status')

# # # Modelling
# =============================================================================

# Classfiers

knn = KNeighborsClassifier()
lgr = LogisticRegression(solver='lbfgs', max_iter=3000)
gnb = GaussianNB()
dt = DecisionTreeClassifier()


# Fidelity tests
def fidelity_tests(model,X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)
    model=Pipeline([("scaler", StandardScaler()), ("model", model)])
    
    model.fit(X_train,y_train) # Fitting train data for predection
    y_pred=model.predict(X_test)
    
    accuracy=accuracy_score(y_test,y_pred) #Accuracy for predection 
    recall=recall_score(y_test,y_pred,average='weighted')# Recall score for predection
    prec=precision_score(y_test,y_pred,average='weighted')# precision score
    conf=confusion_matrix(y_test, y_pred)#predection of late delivery
    f1=f1_score(y_test, y_pred,average='weighted')#f1 score for prediction
    print('Model paramters used are :',model)
    print('Accuracy of late delivery status is:', (accuracy)*100,'%')
    print('Recall score of late delivery status is:', (recall)*100,'%')
    print('Precision score of late delivery status is:', (recall)*100,'%')
    print('Conf Matrix of late delivery status is: \n',(conf))
    print('F1 score of late delivery status is:', (f1)*100,'%')
    
# Cross-validation
def cross_val(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3) 
    model=Pipeline([("scaler", StandardScaler()), ("model", model)])
    model.fit(X_train,y_train)# Fitting train data for predection
    
    cross_model = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)# 10-fold cross-validation

    print("Cross validation model",cross_model)
    print("The average accuracy of model with cross validation is:",cross_model.mean())


train_data=df_order.copy()
# drop the features multicoliniarity/outcome variables
train_data=train_data.drop(['Order Id','Order Customer Id','Order City','Order State','Order Zipcode','order date (DateOrders)', 'Order Item Discount','Order Item Id','Sales','Order Item Total','Days for shipping (real)'], axis=1)

# convert all object data to numerical data
le = preprocessing.LabelEncoder()
train_data['Market'] = le.fit_transform(train_data['Market'])
train_data['Order Country'] = le.fit_transform(train_data['Order Country'])
train_data['Order Region'] = le.fit_transform(train_data['Order Region'])
train_data['Order Status'] = le.fit_transform(train_data['Order Status'])
train_data['Type']   = le.fit_transform(train_data['Type'])
train_data['Delivery Status'] = le.fit_transform(train_data['Delivery Status'])


# # # Prediction of late delivery status (0,1,2,3)
train_data1=train_data.drop('Late Delivery Risk', axis=1)
train_data.info()

# create train/test sets
X1=train_data.loc[:, train_data.columns != 'Delivery Status']
y1=train_data['Delivery Status']

# classfication Results
fidelity_tests(knn,X1,y1)
cross_val(knn,X1,y1)

fidelity_tests(lgr,X1,y1)
cross_val(lgr,X1,y1)

fidelity_tests(gnb,X1,y1)
cross_val(gnb,X1,y1)

fidelity_tests(dt,X1,y1)
cross_val(dt,X1,y1)



# # # Prediction of late delivery risk (0,1)
train_data2=train_data.drop('Delivery Status', axis=1)
train_data.info()

# create train/test sets
X2=train_data2.loc[:, train_data2.columns != 'Late Delivery Risk']
y2=train_data2['Late Delivery Risk']

# classfication Results
fidelity_tests(knn,X2,y2)
cross_val(knn,X2,y2)

fidelity_tests(lgr,X2,y2)
cross_val(lgr,X2,y2)

fidelity_tests(gnb,X2,y2)
cross_val(gnb,X2,y2)

fidelity_tests(dt,X2,y2)
cross_val(dt,X2,y2)

# source: https://www.kaggle.com/nilufarhosseini/supply-chain-data-analysis-99-accuracy
# https://www.kaggle.com/pvmanish/group-9#Casestudy-2:-Late-Delivery-Classification
# https://www.kaggle.com/skloveyyp/comparison-of-classification-regression-rnn
