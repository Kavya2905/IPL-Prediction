import pandas as pd
import numpy as np
import joblib

def predictRuns(input):

    with open ('model.joblib', 'rb') as f:
        regressor = joblib.load(f)
    with open ('venue.joblib', 'rb') as f:
        venue_encoder = joblib.load(f)
    with open ('striker.joblib', 'rb') as f:
        striker_encoder= joblib.load(f)
    with open ('bowler.joblib', 'rb') as f:
        bowler_encoder = joblib.load(f)
    with open ('batteam.joblib', 'rb') as f:
        bat_encoder = joblib.load(f)
    with open ('bowlteam.joblib', 'rb') as f:
        bowl_encoder = joblib.load(f)


    data = pd.read_csv(input)
    data[['striker','non_striker']] = data.batsmen.str.split(",", 1, expand=True)
    data[['bowler', 'non_bowler']] = data.bowlers.str.split(',', 1, expand = True)
    data.drop(labels='non_striker', axis=True, inplace=True)
    data.drop(labels='batsmen', axis=True, inplace=True)
    data.drop(labels='non_bowler', axis=True, inplace=True)
    data.drop(labels='bowlers', axis=True, inplace=True)

    data['venue'] = venue_encoder.transform(data['venue'])
    data['striker'] = striker_encoder.transform(data['striker'])
    data['bowler'] = bowler_encoder.transform(data['bowler'])
    data['batting_team'] = bat_encoder.transform(data['batting_team'])
    data['bowling_team'] = bowl_encoder.transform(data['bowling_team'])

    temp_array = list()


    venue = data['venue']
    innings = data['innings']
    bat = data['batting_team']
    bowl = data['bowling_team']
    striker = data['striker']
    bowler = data['bowler']

    temp_array = temp_array + [venue, innings, bat, bowl, striker, bowler]
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    data = np.array([temp_array])
    data =data.reshape(data.shape[0], -1)
    prediction = int(regressor.predict(data)[0]) + 17

    return prediction




