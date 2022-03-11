import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


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

# delievery status vs type
sns.countplot(x='Type', hue='Delivery Status', data = df_order)

# delievery status vs Market
sns.countplot(x='Market', hue='Delivery Status', data = df_order)

# delievery status vs Country
sns.countplot(x='Order Country', hue='Delivery Status', data = df_order)
