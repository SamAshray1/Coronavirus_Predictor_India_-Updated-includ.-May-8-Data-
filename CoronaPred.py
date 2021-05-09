import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# Finding the first index where India appears, can use for other countries as well.
# Use the below lines to get the index of req. country in the dataset.
# Index_label = dataset[dataset['location'] == "India"].index.tolist()
# print(Index_label[0],Index_label.pop())

# Loading Data, taking only location, total_cases as features.
dataset = pd.read_csv('Covidcases.csv', sep=',')
dataset = dataset.iloc[35737:36202, 2:5]
id = []
for i in range(1,len(dataset["location"])+1):
    id.append(i)
dataset.insert(2, "id", id, True)

# Preparing Data #
x = np.array(dataset['id']).reshape(-1, 1)
y = np.array(dataset['total_cases']).reshape(-1, 1)
#30plt.plot(y,'-r')
# plt.show()

polyFeat = PolynomialFeatures(degree=7)
x = polyFeat.fit_transform(x)
#print(x)

# Training Model on Data
regressor = linear_model.LinearRegression()
regressor.fit(x,y)
accuracy = regressor.score(x, y)
print(f'Accuracy:', round(accuracy*100, 3))
y_pred = regressor.predict(x)
#plt.plot(y_pred, '--b')
#plt.show()

# Future Prediction
days = input("How many days after May 8th,2021; Do you need the prediction of total cases for?\n")
print('-'*20)
# To find the number of days
#print(id[len(id)-1])
print('Prediction - Cases after', days, "day:")
print(int(regressor.predict(polyFeat.fit_transform([[465+int(days)]]))))

x1 = np.array(list(range(1, 465+int(days)))).reshape(-1, 1)
y1 = regressor.predict(polyFeat.fit_transform(x1))
plt.plot(y1, '--r')
plt.plot(y_pred, '--b')
plt.show()