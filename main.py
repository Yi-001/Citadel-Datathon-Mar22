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

# overview
df_order.info()
df_order.head()
df_order.describe()

# check na
df_order.isnull().values.any()

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

# delivery status prediction
# need to do a PCA first?
train_data=df_order.copy()
train_data=train_data.drop(['Order Id','Order Customer Id','Order City','Order State','Order Zipcode','order date (DateOrders)', 'Order Item Discount','Order Item Id','Sales','Order Item Total','Days for shipping (real)','Late Delivery Risk'], axis=1)

train_data.info()
le = preprocessing.LabelEncoder()
train_data['Market'] = le.fit_transform(train_data['Market'])
train_data['Order Country'] = le.fit_transform(train_data['Order Country'])
train_data['Order Region'] = le.fit_transform(train_data['Order Region'])
train_data['Order Status'] = le.fit_transform(train_data['Order Status'])
train_data['Type']   = le.fit_transform(train_data['Type'])
train_data['Delivery Status'] = le.fit_transform(train_data['Delivery Status'])

train_data.head()
x=train_data.loc[:, train_data.columns != 'Delivery Status']
y=train_data['Delivery Status']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
knn_pipline = Pipeline([("scaler", StandardScaler()), ("KNeighborsClassifier", KNeighborsClassifier())])
model = knn_pipline.fit(X_train, y_train)
knn_train_predict = pd.DataFrame({'actual' : y_train,'predicted' : model.predict(X_train)})
knn_train_predict.head()
