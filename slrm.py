# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# load dataset
db=pd.read_csv(r"C:\Users\SHRUTHI\OneDrive\Desktop\FSDS\Assignments\projects\projects to do\pj-12 hpp, slr&mlr\Salary_Data.csv")
# Feature selection (independent variable X and dependent variable y)
x=db.iloc[:,:-1] # Years of experience (Independent variable)

y=db.iloc[:,-1]  # Salary (Dependent variable)

# Split the dataset into training and testing sets (80% training, 20% testing)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.20, random_state=0)

# Reshape x_train and x_test into 2D arrays if they are single feature columns
x_train = x_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)

# You don't need to reshape y_train, as it's the target variable
# Fit the Linear Regression model to the training set
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train, y_train)

# Predicting the results for the test set
y_pred=regressor.predict(x_test)

# Compare predicted and actual salaries from the test set
comparision =pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparision)

# Optional: Output the coefficients of the linear model
m_slope= regressor.coef_
m_slope 

c_intercept=regressor.intercept_
c_intercept

#Visualizing the Training set results
plt.scatter(x_train, y_train, color='red') # real salary data (training)
plt.plot(x_train, regressor.predict(x_train),color='blue') # Predicted regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the Test set results
plt.scatter(x_test, y_test, color='red') # real salary data (training)
plt.plot(x_train, regressor.predict(x_train),color='blue') # Predicted regression line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary for 12 and 20 years of experience using the trained model
y_12=regressor.predict([[12]])
y_20=regressor.predict([[20]])
print(F'Predicted salary for 12 years and 20 years of experience are {y_12[0]} and {y_20[0]}')

from sklearn.metrics import mean_squared_error
# check model performance
bias = regressor.score(x_train, y_train)
variance=regressor.score(x_test,y_test)
train_mse=mean_squared_error(y_train, regressor.predict(x_train))
test_mse=mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
import pickle
filename='lrm.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor, file)
print('Model has been pickled and saved as lrm.pkl')