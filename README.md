# TSF-Task-1
# This model predicts the marks of a student based on the no of hours studied.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

url="http://bit.ly/w-data"
s_data=pd.read_csv(url)
print("Data imported successfully")
s_data.head(10)

s_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Pecentage score')
plt.show()

X=s_data.iloc[:, :-1].values
y=s_data.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
print("Training Complete.")

line = regressor.coef_*X+regressor.intercept_
plt.scatter(X,y)
plt.plot(X,line)
plt.show()

print(X_test)
y_pred=regressor.predict(X_test)

df=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
df

hours = 9.25
#X_pred = regressor.predict([hours])
print("No of Hours = {}".format(hours))
print("Predicted Score = ",regressor.predict([[hours]]))

from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))
