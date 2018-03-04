from starter import *
import pickle
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

weather_grouped = pickle.load(open('weather_grouped.p', 'rb'))
holidays = pickle.load(open('holidays.p', 'rb'))

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
#     if dropnan:
#         agg.dropna(inplace=True)
    return agg


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
    reframed.drop(reframed.columns[[reframed.shape[1] - i for i in range(1,data.shape[1])]], axis=1, inplace=True)
    print(reframed.head())
    return reframed, scaler

# concatenate train and test sets
def load_site(train=None, test=None, site_id=None, weather=weather_grouped, holidays=holidays):
    train = train[train.SiteId==site_id]
    test = test[test.SiteId==site_id]
    train.loc[:,'ForecastId'] = 0
    data = pd.concat([train, test], axis=0)
    data = data.merge(weather, how='left', on=['Timestamp', 'SiteId'])
    data = data.merge(holidays, how='left', on=['Date','SiteId'])
    data['Holiday'].fillna(0, inplace=True)
    data['isHoliday'].fillna(0, inplace=True)
    data.drop(['Unnamed: 0_x','Unnamed: 0_y'], axis=1, inplace=True)
    return data

# find start and end times for site
def find_start_end(data):
    time_pairs = []
    forecast_ids = data.ForecastId.unique()[1:]
    forecast_length = data[data.ForecastId==forecast_ids[0]].shape[0]-1
    timedelta = data.Timestamp[1] - data.Timestamp[0]
    train_start = data.Timestamp[0]
    test_end = data[data.ForecastId==forecast_ids[0]].Timestamp[forecast_length]
    test_start = test_end - forecast_length*timedelta
    train_end = test_start - timedelta
    time_pairs.append((train_start, train_end, test_start, test_end))
    
    for forecast_id in forecast_ids[1:]:
        forecast_length = data[data.ForecastId==forecast_id].shape[0]-1
        train_start = test_end + timedelta
        test_end = data[data.ForecastId==forecast_id].Timestamp[forecast_length]
        test_start = test_end - forecast_length*timedelta
        train_end = test_start - timedelta
        time_pairs.append((train_start, train_end, test_start, test_end))
    
    return time_pairs

# segment and interpolate data for site
def prepare_data(data=None, features= ['Value', 'Temperature'], method='linear'):
    data.index = data.Timestamp
    data = data[features]
    data['Temperature'].interpolate(method, inplace=True)
    try:
        # start where there isn't consecutive NaNs
        start_index = data[(data.Temperature.isnull()==False)&(data.Temperature.shift(-1).isnull()==False)].index[0]
        data = data.loc[start_index:,:]
    except:
        print('No temperature data')
    return data

# split train/test sets from reframed data
def split_train_test(reframed, n_train_days):
    values = reframed.values
    train = values[:n_train_days, :]
    test = values[n_train_days+1:, :]
    # Split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # Reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y


# build multivariate LSTM model for predictions
def create_LSTM(train_X=None, train_y=None, epochs=50, batch_size=100):
    model = Sequential()
    model.add(LSTM(4, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=epochs,
                                batch_size=batch_size,
                                verbose=1, shuffle=False)
    return model

# make predictions with LSTM model
def model_predict(model, test_X, n_features, scaler):
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    inv_yhat = np.concatenate([yhat, test_X], axis=1)[:,:n_features]
    inv_yhat = scaler.inverse_transform(inv_yhat)[:,0]
    return inv_yhat

#adds prediction to proper submission format
def predict_site(train=None, test=None, site_id=None):
    #load site data and train/test times
    data = load_site(train, test, site_id)
    data.sort_values('Timestamp', inplace=True)
    data.index = data.Timestamp
    data = prepare_data(data, features=['Value', 'Temperature','isHoliday','ForecastId','Timestamp'])
    time_pairs = find_start_end(data)
    
    #build model and predict for each forecastid
    for n, times in enumerate(time_pairs):
        df = data[(data.Timestamp >= time_pairs[n][0]) & (data.Timestamp <= time_pairs[n][3])]
        df = prepare_data(df, features=['Value','Temperature','isHoliday','ForecastId'])
        df['Temperature'].fillna(0, inplace=True)
        df.loc[:,'Timestamp'] = df.index
        df.drop(['ForecastId','Timestamp'],axis=1, inplace=True)
        n_train_days = df[(df.index >=time_pairs[n][0]) & (df.index <=time_pairs[n][1])].shape[0]-1
        n_features = df.shape[1]
        reframed_data, scaler = reframe_features(df, 2)
        reframed_data.fillna(0, inplace=True)
        train_X, train_y, test_X, test_y = split_train_test(reframed_data, n_train_days)
        model = create_LSTM(train_X, train_y, epochs=50)
        inv_yhat = model_predict(model, test_X, n_features, scaler)
        data.loc[time_pairs[n][2]:time_pairs[n][3], 'Value'] = inv_yhat
        
    return data



