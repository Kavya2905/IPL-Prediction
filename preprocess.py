import numpy as np
import glob
import pandas as pd
from sklearn import preprocessing
import joblib

df = pd.read_csv('all_matches.csv')
df['batsmen'] = df[['striker', 'non_striker']].apply(lambda x: ', '.join(x), axis = 1)

venue_encoder = preprocessing.LabelEncoder()
striker_encoder = preprocessing.LabelEncoder()
bowler_encoder = preprocessing.LabelEncoder()
bat_encoder = preprocessing.LabelEncoder()
bowl_encoder = preprocessing.LabelEncoder()

df['venue']= venue_encoder.fit_transform(df['venue'])

df['striker']= striker_encoder.fit_transform(df['striker'])

df['bowler']= bowler_encoder.fit_transform(df['bowler'])


consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                                'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                                'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['batting_team'].isin(consistent_teams)) & (df['bowling_team'].isin(consistent_teams))]

df = df[df['ball'] <= 5.6]
df = df[df['innings'] <= 2]
df=df.fillna(0)
df['runs'] = df['runs_off_bat'] + df['extras'] + df['wides'] + df['noballs'] + df['byes'] + df['legbyes'] + df['penalty']

df['total_runs'] = df.groupby(df.index // 36)['runs'].transform('sum')

df['batting_team'] = bat_encoder.fit_transform(df['batting_team'])

df['bowling_team'] = bowl_encoder.fit_transform(df['bowling_team'])

from datetime import datetime
df['start_date'] = df['start_date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))

df = df.reset_index()
           # --- Data Preprocessing ---
           # Converting categorical features using OneHotEncoding method


           # Rearranging the columns
encoded_df = df[['start_date', 'venue','innings','batting_team', 'bowling_team', 'striker', 'bowler', 'total_runs']]


joblib.dump(venue_encoder, 'venue.joblib')
joblib.dump(striker_encoder, 'striker.joblib')
joblib.dump(bowler_encoder, 'bowler.joblib')
joblib.dump(bat_encoder, 'batteam.joblib')
joblib.dump(bowl_encoder, 'bowlteam.joblib')

encoded_df.to_csv('trial.csv')

print(encoded_df)