from starter import *
import pickle
import datetime
from lstm_script import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

submission_frequency = pickle.load(open('submission_frequency.p', 'rb'))
holidays = pickle.load(open('holidays.p', 'rb'))
weather_grouped = pickle.load(open('weather_grouped.p', 'rb'))
train = pickle.load(open('train.p', 'rb'))
submission_format = pickle.load(open('submission_format.p', 'rb'))
official_submission = pd.read_csv('submission_format.csv')

submission_union = submission_format.groupby(['ForecastId']).mean()
submission_union['ForecastId'] = submission_union.index
submission_frequency = submission_frequency.merge(submission_union[['SiteId','ForecastId']], how='left', on='ForecastId')

def prepare_submission(train=None, test=None, submission=None, 
                       sites=submission_frequency.SiteId.unique()):
    
    #convert Timestamp to datetime
    submission.loc[:,'Timestamp'] = pd.to_datetime(submission.loc[:,'Timestamp'])
    submission.index= submission['Timestamp']
    
    for site in sites:
        print('Making predictions for Site {}'.format(site))
        predictions = predict_site(train, test, site)
        submission.loc[submission.SiteId==site,'Value'] = predictions[predictions.ForecastId>0]['Value'].values
        print('Finished predictions for Site {}'.format(site))
    return submission


if __name__ == '__main__':
	submission = prepare_submission(train, submission_format, official_submission)
	submission.to_csv('submission.csv')