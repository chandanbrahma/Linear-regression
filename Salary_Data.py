## importing data
import pandas as pd
data= pd.read_csv('E:\\assignment\\simplelinearregression\\Salary_Data.csv')
data.head()
data.info()
data.describe()


## so we do have 2 columns with 30 rows and we need to predict the salary hike

import matplotlib.pyplot as plt
##ploting the dependent variable
plt.hist(data['Salary'])
plt.boxplot(data['Salary'])

##ploting the independent variable
plt.hist(data['YearsExperience'])
plt.boxplot(data['YearsExperience'])

## plot between x and y
plt.plot(data['YearsExperience'],data['Salary'],"ro");plt.xlabel("YearsExperience");plt.ylabel("Salary")
## from the plot we can visualize the coorelation between the data

data.corr()
## so there is an excellent corelation of 0.978 between both the variables

##importing the statsmodels.formula.api and using ols technique for calculating best fit line
import statsmodels.formula.api as smf

model=smf.ols('Salary~YearsExperience',data=data).fit()

# P-values for the variables and R-squared value for prepared model
model.summary()

## so we got a R^2 value of 0.957 which is great, now lets predict the values and plot the graph

prediction=model.predict(data)

##visualisation
import matplotlib.pyplot as plt
plt.title('Comparison of years of experience and prediction salary')
plt.ylabel('Predicted values')
plt.xlabel('Years of exp')
plt.plot(data['YearsExperience'], prediction, color='blue', linewidth=3)
plt.show()
