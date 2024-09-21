import numpy as np
import pandas as pd


# Importing the datasets

dataset = pd.read_csv("C:/Users/Ankit Singh/Desktop/Heart Disease Prediciton Using ANN/database/heart.csv")


# Since RowNumber,CustomerId And Surname doesn't provide much information on predicting the customer churning behaviour in a bank
# So, we remove those columns in our dataset

x = dataset.iloc[ : , : 11].values
y = dataset.iloc[ : , 11].values

print(x)
# By observing the feature values ,we know that country and gender are categorical values in the dataset and 
#  while building our machine learning models the categorical varaibale and values are not allowed. So we need to encode those categorical data

# Encoding Categorical Data

 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Now Data preprocessing step is finished now we must focus on building the architecture of ANN

#Importing the Keras Libraries and Packages

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
import tensorflow as tf

rms_model = Sequential()

rms_model.add(Dense(units=16,kernel_initializer='uniform',activation='relu',input_dim=11))
rms_model.add(Dense(units = 16,kernel_initializer='uniform',activation='relu'))
rms_model.add(Dense(units = 8,kernel_initializer='uniform',activation='relu'))
rms_model.add(Dense(units = 16,kernel_initializer='uniform',activation='relu'))
rms_model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
rms_model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
rms_model.fit(x=X_train,y=y_train,batch_size=25,epochs=150,validation_data=(X_test,y_test))

# From all the computations we can say that rms prop optimizer perform better than adam optimizer. So we the final model has 16 hidden layers with batch 
# size of 25 and optimizer equals adam and the epochs is equal to 200 and the average validation and testing accuracy is 86%

# Saving our model and it's architecture
# Since rms prop is performing well so we serailize rms_prop model to use it further for deployment.
import json
# serialize model to JSON
model_json = rms_model.to_json()
with open("heart_diseases.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
rms_model.save_weights("heart_diseases.h5")
print("Saved model to disk")


