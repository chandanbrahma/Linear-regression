## importing data
import pandas as pd 
data= pd.read_csv('E:\\assignment\\simplelinearregression\\calories_consumed (1).csv')
data.head()
data.info()
data.describe()


## so we do have 2 columns with 14 rows and we need to calculate the weight gained based on the calories consumed

import matplotlib.pyplot as plt
##ploting the dependent variable
plt.hist(data['Weight gained (grams)'])
plt.boxplot(data['Weight gained (grams)'])

##ploting the independent variable
plt.hist(data['Calories Consumed'])
plt.boxplot(data['Calories Consumed'])

## plot between x and y
plt.plot(data['Calories Consumed'],data['Weight gained (grams)'],"ro");plt.xlabel("Calories Consumed");plt.ylabel("Weight gained (grams)")
## from the plot we can visualize there is a good coorelation between the data


##checking th corelation between the data
data.corr()
##we can see there is a corelation of 0.946 which is good corelation

##importing the statsmodels.formula.api and using ols technique for calculating best fit line

import statsmodels.formula.api as smf
a=data['Calories Consumed']
b=data['Weight gained (grams)']
model=smf.ols('b~a',data=data).fit()

# P-values for the variables and R-squared value for prepared model
model.summary()

## so we got a r^2 value of 0.897
prediction = model.predict(data)



##scatter plot 
import matplotlib.pyplot as plt
plt.title('Comparison of calories consumed and the Predicted values')
plt.ylabel('Predicted values')
plt.xlabel('Calories Consumed')
plt.plot(data['Calories Consumed'], prediction, color='blue', linewidth=3)

