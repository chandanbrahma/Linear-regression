## importing data 
import pandas as pd
import numpy as np
data= pd.read_csv('E:\\assignment\\simplelinearregression\\delivery_time.csv')
data.head()
data.info()
data.describe()

## so we do have 2 columns with 21 rows and we need to predict the delivery time using the sorting time

import matplotlib.pyplot as plt
##ploting the dependent variable
plt.hist(data['Delivery Time'])
plt.boxplot(data['Delivery Time'])

##ploting the independent variable
plt.hist(data['Sorting Time'])
plt.boxplot(data['Sorting Time'])

## plot between x and y
plt.plot(data['Sorting Time'],data['Delivery Time'],"ro");plt.xlabel("Sorting Time");plt.ylabel("Delivery Time")
## from the plot we can visualize there is a some corelation between the data.
##now lets calculate the value

data.corr()
##the corelation is 0.82 which is a good corelation.

##importing the statsmodels.formula.api and using ols technique for calculating best fit line

import statsmodels.formula.api as smf
a=data['Sorting Time']
b=data['Delivery Time']
model=smf.ols('b~a',data=data).fit()

# P-values for the variables and R-squared value for prepared model
model.summary()

## so we got a r^2 value of 0.682

## as the value of r^2 is not that good, so lets try to improve the same
##applying logarithemic approach
model2 = smf.ols('a~np.log(b)',data=data).fit()

model2.summary()

## now the r^2 value is increased to 0.711

model3 = smf.ols('np.log(a)~b',data=data).fit()

model3.summary()

##R^2 is 0.695

## lets try for quadratic approach
data["c"] = data['Delivery Time'] * data['Delivery Time']
model_quad = smf.ols("a~b+c",data=data).fit()

model_quad.summary() 

## as we have trialed all the methods , and here we got a R^2 value of 0.716
prediction = model_quad.predict(data)
##visualization
import matplotlib.pyplot as plt
plt.title('Comparison of sorting time and the Predicted delivery time')
plt.ylabel('Predicted delivery time')
plt.xlabel('sort time')
plt.plot(data['Sorting Time'], prediction, color='blue', linewidth=2)
plt.show()
