## importing data
import pandas as pd 
data= pd.read_csv('E:\\assignment\\simplelinearregression\\emp_data.csv')
data.head()
data.info()
data.describe()


## so we do have 2 columns with 10 rows and we need to predict the churn_out rate

import matplotlib.pyplot as plt
##ploting the dependent variable
plt.hist(data['rate'])
plt.boxplot(data['rate'])

##ploting the independent variable
plt.hist(data['hike'])
plt.boxplot(data['hike'])

## plot between x and y
plt.plot(data['hike'],data['rate'],"ro");plt.xlabel("hike");plt.ylabel("rate")
## from the plot we can visualize the coorelation between the data

data.corr()
## so we found a negaive corelation of 0.911

import statsmodels.formula.api as smf

model=smf.ols('rate~hike',data=data).fit()

# P-values for the variables and R-squared value for prepared model
model.summary()

## so we got a good R^2 value of 0.831, now lets predict the values
prediction=model.predict(data)

##visualisation
import matplotlib.pyplot as plt
plt.title('Comparison of salary hike and prediction churan_out rate')
plt.ylabel('Predicted values')
plt.xlabel('salary hike')
plt.plot(data['hike'], prediction, color='blue', linewidth=3)
plt.show()
