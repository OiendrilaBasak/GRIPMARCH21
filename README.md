# GRIPMARCH21
Data Science and Business Analytics Internship
#    Name: Oiendrila Basak

##     Data Science and Business Analytics intern at The Sparks Foundation

# Task 1

### **Simple Linear Regression**
In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.


#importing all the libraries required in this notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#Reading data from remote link
url="http://bit.ly/w-data"
s_data=pd.read_csv(url)
print("Data imported succesfully")

s_data.head(10)

Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

#Plotting the distribution of scores
s_data.plot(x='Hours',y='Scores',style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours studied')
plt.ylabel('Percentage score')
plt.show()

**From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

### **Preparing the data**

The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

X=s_data.iloc[:, :-1].values
Y=s_data.iloc[:, 1].values

Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

### **Training the Algorithm**
We have split our data into training and testing sets, and now is finally the time to train our algorithm. 

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
print("Training complete")

#Plotting the regression
line=regressor.coef_*X+regressor.intercept_

#Plotting for the test data
plt.scatter(X,Y)
plt.plot(X,line)
plt.show()


### **Making Predictions**
Now that we have trained our algorithm, it's time to make some predictions.

print(X_test)      #to print testing data in hours
Y_pred=regressor.predict(X_test)  #Predicting the score

#Comparing actual vs predicted values
df=pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred})
df

#To test with my own data
array=np.array(9.25)
hours=array.reshape(-1,1)
p=regressor.predict(hours)
prediction=p.ravel()[0]
print("No of hours={}".format(hours.ravel()[0]))
print("Predicted Scores={}".format(prediction))

### **Evaluating the model**

The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

from sklearn import metrics
print('Mean absolute Error:',metrics.mean_absolute_error(Y_test,Y_pred))
