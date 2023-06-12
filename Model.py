import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import pickle
import warnings
warnings.filterwarnings('ignore')
a=pd.read_csv("Gait_Data.csv")
print(a)


############################ features  ####################################
X=a.drop(['Activity'],axis=1)

print(X)
############################   labels  ######################################
Y=a['Activity']
print(Y)
############################# traing and testing part #######################
x_train,x_test,y_train,y_test = train_test_split(X,Y,shuffle=True,test_size=0.25, random_state=0)


############################# Algorithm Implementation #######################


############################# Gaussian Naive Bayes  #######################
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(x_train, y_train)  #train the data
y_pred=NB.predict(x_test)
##print(y_pred)
##print(y_test)
print('Naive Bayes ACCURACY is', accuracy_score(y_test,y_pred))

############################# RandomForestClassifier  #######################
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)  #train the data
y_pred_rf=clf.predict(x_test)
##print(y_pred)
##print(y_test)
print('Random Forest ACCURACY is', accuracy_score(y_test,y_pred_rf))


############################# DecisionTreeClassifier  #######################
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(random_state=0)
DT.fit(x_train, y_train)  #train the data
y_pred_DT=DT.predict(x_test)
##print(y_pred)
##print(y_test)
print('DecisionTree ACCURACY is', accuracy_score(y_test,y_pred_DT))
############################  LSTM  #############################

from keras.models import Sequential
from keras.layers import LSTM, Dense
model_LSTM = Sequential()
model_LSTM.add(LSTM(32, input_shape=(12,1)))
model_LSTM.add(Dense(16, activation='relu'))
model_LSTM.add(Dense(1, activation='sigmoid'))

# Compile the model
model_LSTM.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model_LSTM.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model_LSTM.evaluate(x_train, y_test)
##print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

##filename = 'model.pkl'
##pickle.dump(DT, open(filename, 'wb'))

