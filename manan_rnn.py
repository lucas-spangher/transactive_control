#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import seed

seed(0)
#%%
data = pd.read_csv("./HourlyDataCleanedFinal.csv", parse_dates=True, index_col="Date",)

# we first try to run the script with no null values for the hourly energy
#%%
data = data.dropna(subset=["HourlyEnergy"])
data

#%%
# we drop columns that are not needed
data.drop(
    ["WorkGroup", "TotalEnenergy", "Name", "Treatment_Administered_Indicator",]
)
#%%
# Data scaling
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Data splitting
np.random.seed(7)

train_len = int(len(data_scaled) * 0.70)
test_len = len(data_scaled) - train_len
train_data = data_scaled[0:train_len, :]
test_data = data_scaled[train_len : len(data_scaled), :]

print(len(train_data), len(test_data))


def dataset_creation(dataset, time_step=1):
    data_x, data_y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i : (i + time_step), 0]
        data_x.append(a)
        data_y.append(dataset[i + time_step, 0])
    return np.array(data_x), np.array(data_y)


time_step = 1
train_x, train_y = dataset_creation(train_data, time_step)
test_x, test_y = dataset_creation(test_data, time_step)

train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

# Keras LSTM model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

model = Sequential()
model.add(LSTM(256, input_shape=(1, time_step)))
# repeat line for bilayer LSTM
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])
model.fit(train_x, train_y, epochs=1, batch_size=1, verbose=1)
model.summary()

score = model.evaluate(train_x, train_y, verbose=0)
print("Keras Model Loss = ", score[0])
print("Keras Model Accuracy = ", score[1])

train_pred = model.predict(train_x)
train_pred = model.predict(test_x)

train_pred = scaler.inverse_transform(train_pred)
train_y = scaler.inverse_transform([train_y])

train_pred = scaler.inverse_transform(train_pred)
test_y = scaler.inverse_transform([test_y])

train_predict_plot = np.empty_like(data_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[1 : len(train_pred) + 1, :] = train_pred

train_predict_plot = np.empty_like(data_scaled)
train_predict_plot[:, :] = np.nan
train_predict_plot[len(train_pred) + (1 * 2) + 1 : len(data_scaled) - 1, :] = train_pred

plt.figure(figsize=(10, 5))
plt.plot(scaler.inverse_transform(data_scaled))
plt.plot(train_predict_plot)
plt.plot(train_predict_plot)
plt.show()


# %%
