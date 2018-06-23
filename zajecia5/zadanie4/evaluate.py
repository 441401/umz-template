from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from keras import optimizers


r = pd.read_csv(os.path.join("train", "train.tsv"), header=None, names=["Price", "IsNew", "Rooms", "Floor", "Location", "SqrMeters"], sep='\t')
X_train = pd.DataFrame(r, columns=["IsNew", "Rooms", "Floor", "SqrMeters"])
Y_train = pd.DataFrame(r, columns=["Price"])


def create_baseline():
    model = Sequential()
    
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu', kernel_initializer='normal')) 
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='zeros'))

    model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['accuracy'])

    return model


estimator = KerasRegressor(build_fn=create_baseline, epochs=100, verbose=True, validation_split = 0.3)

#scaler = StandardScaler()
#scaler.fit(X_train)
estimator.fit(X_train, Y_train)
predictions_train = estimator.predict(X_train)



#DEV
r = pd.read_csv(os.path.join("dev-0", "in.tsv"), header=None, names=["IsNew", "Rooms", "Floor", "Location", "SqrMeters"], sep='\t')
X_dev = pd.DataFrame(r, columns=["IsNew", "Rooms", "Floor", "SqrMeters"])
Y_dev = pd.read_csv(os.path.join("dev-0", "expected.tsv"), header=None, names=["Price"], sep='\t')

#scaler.fit(X_dev)
predictions_dev = estimator.predict(X_dev)

with open(os.path.join("dev-0", "out.tsv"), 'w') as file:
    for prediction in predictions_dev:
        file.write(str(prediction) + '\n')

#MASTER
r = pd.read_csv(os.path.join("dev-0", "in.tsv"), header=None, names=["IsNew", "Rooms", "Floor", "Location", "SqrMeters"], sep='\t')
X_master = pd.DataFrame(r, columns=["IsNew", "Rooms", "Floor", "SqrMeters"])

#scaler.fit(X_master)
predictions_master = estimator.predict(X_master)

with open(os.path.join("test-A", "out.tsv"), 'w') as file:
    for prediction in predictions_master:
        file.write(str(prediction) + '\n')
