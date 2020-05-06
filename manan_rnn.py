#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt

# %%
df = pd.read_csv(
    "./new_action_energy_data/0_extra_train/more_data_action_energy_linear_extratrain_0.csv"
)
# %%
cbh = pd.offsets.CustomBusinessHour(start="08:00", end="18:00")
df["Timestamp"] = pd.date_range(
    start=pd.Timestamp("2018-09-20T08"), freq=cbh, periods=len(df)
)


# %%
df = df.drop(columns={"Unnamed: 0", "Hour"})
df = df.set_index("Timestamp")

# %%
df = df.dropna(axis=0)

# %%
""" Reference: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/ """


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# %%
values = df.values
scaler = MinMaxScaler(feature_range=(0, 1))
values = values.astype("float32")
print(values)
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 5, 1)

# %%
reframed = reframed.drop(reframed.columns[[-2]], axis=1)

# %%
""" Let's do a 70:30 training test split """
values_reframed = reframed.values
train_margin = int(len(values_reframed) * 0.7)
train = values_reframed[:train_margin, :]
test = values_reframed[train_margin:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# %%
# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss="mae", optimizer="adam")
# fit network
history = model.fit(
    train_X,
    train_y,
    epochs=50,
    batch_size=72,
    validation_data=(test_X, test_y),
    verbose=2,
    shuffle=False,
)
# plot history
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="test")
plt.legend()
plt.show()


# %%
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# %%
# invert scaling for forecast
inv_yhat = np.concatenate((test_X[:, 0][:, np.newaxis], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]
#%%
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X[:, 0][:, np.newaxis], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print("Test RMSE: %.3f" % rmse)

# %%
