import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_csv('challenge_dataset.txt')
x_values = dataframe[['x']]
y_values = dataframe[['y']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#get errors and print
error = body_reg.predict(x_values.values)-y_values.values
mean_absolute_error = np.mean(error)
standard_deviation = np.std(error)
print ('Mean squared error:{:.2f}, standard deviation: {:.2f}'.format(mean_absolute_error, standard_deviation))

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values),c=('m'), )
plt.show()
