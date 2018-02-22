from starter import *
import pickle
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM


def RMSLE(y, pred):
    return mean_squared_error(y, pred)**0.5


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# acquire siteid data
def get_site_data(data=None, SiteId=None, features=['Value','Temperature']):
    data = data[data.SiteId==SiteId]
    data.index = data.Timestamp
    data = data[features]
    return data

# normalize and reframe features
def reframe_features(data=None, n_in=1):
    values = data.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_in, 1)
    reframed.drop(reframed.columns[[reframed.shape[1] - i for i in range(1,data.shape[1])]], 
        axis=1, inplace=True)
    print(reframed.head())
    return reframed

# split train/test sets from reframed data
def split_train_test(reframed):
    values = reframed.values
    n_train_days = int(len(values) * 0.7)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # Split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # Reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y

# build multivariate LSTM model
def create_LSTM(train_X=None, train_y=None, test_X=None, test_y=None, epochs=50, batch_size=100):
    model = Sequential()
    model.add(LSTM(4, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=epochs,
                                batch_size=batch_size, validation_data=(test_X, test_y),
                                verbose=1, shuffle=False)
    return model

# compare forecast to actual
def forecast_score(pred, actual_input, actual_output):
    actual_input = actual_input.reshape((actual_input.shape[0], actual_input.shape[2]))
    
    # Invert scaling for forecast
    inv_pred = np.concatenate((pred, actual_input[:, 1:]), axis=1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(inv_pred)
    inv_pred = scaler.inverse_transform(inv_pred)
    inv_pred = inv_pred[:,0]
    
    # Invert scaling for actual
    actual_output = actual_output.reshape((len(actual_output), 1))
    inv_output = np.concatenate((actual_output, actual_input[:, 1:]), axis=1)
    inv_output = scaler.inverse_transform(inv_output)
    inv_output = inv_output[:,0]
    rmsle = RMSLE(inv_output, inv_pred)
    
    return rmsle



if __name__ == '__main__':

	# load data
	submission_frequency = pickle.load(open('submission_frequency.p', 'rb'))
	metadata = pickle.load(open('metadata.p', 'rb'))
	df = pickle.load(open('df.p', 'rb'))

	# clean data
	df.fillna(0, inplace=True)
	df.drop(['Unnamed: 0_x','Unnamed: 0_y'], axis=1, inplace=True)

	df1 = get_site_data(df, 1, ['Value','Temperature','isHoliday'])
	df1['Temperature'].interpolate('linear', inplace=True)
	reframed_data = reframe_features(df1, 1)
	train_X, train_y, test_X, test_y = split_train_test(reframed_data)
	multi_model = create_LSTM(train_X, train_y, test_X, test_y)
	yhat = multi_model.predict(test_X)
	print(forecast_score(yhat, test_X, test_y))

