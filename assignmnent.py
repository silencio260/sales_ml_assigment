import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

# importing datasets
data_set = pd.read_csv('vgsales.csv')

df_nintendo = data_set[data_set["Publisher"].str.contains("Nintendo") == False]

print(df_nintendo['Publisher'])
print(df_nintendo['Global_Sales'].max())

##
df_2 = data_set.groupby(['Year', 'Genre'], as_index=False)['Global_Sales'].sum()
#df_avg = data_set.groupby(['Year', 'Genre'], as_index=False)['Global_Sales'].mean()
print(df_2)

########
nintendo = data_set[data_set["Publisher"].str.contains("Nintendo") == True]
nintendo['Japan_EU'] = nintendo['JP_Sales'] + nintendo['EU_Sales']
nt_final = nintendo.groupby(['Year'], as_index=False)['Japan_EU'].sum()
#print(nt_final)

########
p_df =  data_set.groupby(['Year', 'Genre'], as_index=False)
#print(p_df)

#####
nt_x = nt_final['Year']
nt_y = nt_final['Japan_EU']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(nt_x,nt_y,test_size=1/3,random_state=0)


from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
#regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
#print("--Prediction Salary---")
#print(y_pred)
#print("---Real Salary----")
#print(y_test)


#####
df_2['Genre'] = df_2['Genre'].astype('category')
df_2['Genre'] = df_2['Genre'].cat.codes
x = df_2.drop(columns = 'Global_Sales')
y = df_2['Global_Sales']
X_train2, X_test2, y_train2, y_test2 = train_test_split(x,y,test_size=1/3,random_state=0)
classifier = LinearRegression()
classifier.fit(X_train2,y_train2)

#y_pred2 = regressor.predict(X_test2)
#print("--Prediction Salary---")
#print(y_pred2)
#print("---Real Salary----")
#print(y_test2)
#print(y)