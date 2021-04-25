import numpy as np
import glob
import pandas as pd

# Loading the dataset
import os
# --- Data Cleaning ---
# Removing unwanted columns
os.chdir("ipl_csv2")
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
df = pd.read_csv("combined_csv.csv")

df=df.fillna(0)

df.insert(22, 'runs', '')

df['runs'] = df['runs_off_bat'] + df['extras'] + df['wides'] + df['noballs'] + df['byes'] + df['legbyes'] + df['penalty']

df.insert(23, 'total_runs', '')

consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                                'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                                'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['batting_team'].isin(consistent_teams)) & (df['bowling_team'].isin(consistent_teams))]

df = df[df['ball'] <= 5.6]


df = df[df['innings'] < 2]

total = 0
for row in df:
    total += int(row[22])
    df['total_runs'] = total

from datetime import datetime
df['start_date'] = df['start_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))

           # --- Data Preprocessing ---
           # Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['batting_team', 'bowling_team'])

           # Rearranging the columns
encoded_df = encoded_df[['start_date','innings', 'ball','batting_team_Chennai Super Kings',
                          'batting_team_Delhi Daredevils', 'batting_team_Kings XI Punjab',
                          'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians', 'batting_team_Rajasthan Royals',
                          'batting_team_Royal Challengers Bangalore', 'batting_team_Sunrisers Hyderabad',
                          'bowling_team_Chennai Super Kings', 'bowling_team_Delhi Daredevils', 'bowling_team_Kings XI Punjab',
                          'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians', 'bowling_team_Rajasthan Royals',
                          'bowling_team_Royal Challengers Bangalore', 'bowling_team_Sunrisers Hyderabad',
                        'runs_off_bat', 'extras','wides','noballs',
                          'byes', 'legbyes','penalty', 'runs']]
X_train = encoded_df.drop(labels='total_runs', axis=1)[encoded_df['start_date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total_runs', axis=1)[encoded_df['start_date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['start_date'].dt.year <= 2016]['total_runs'].values
y_test = encoded_df[encoded_df['start_date'].dt.year >= 2017]['total_runs'].values

           # Removing the 'date' column

X_train.drop(labels='start_date', axis=True, inplace=True)
X_test.drop(labels='start_date', axis=True, inplace=True)

# --- Model Building ---
           # Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

temp_array = list()

bat_team = input('Enter name of Batting team:')
if bat_team == 'Chennai Super Kings':
    temp_array = temp_array + [1, 0, 0, 0, 0, 0, 0, 0]
elif bat_team == 'Delhi Daredevils':
    temp_array = temp_array + [0, 1, 0, 0, 0, 0, 0, 0]
elif bat_team == 'Kings XI Punjab':
    temp_array = temp_array + [0, 0, 1, 0, 0, 0, 0, 0]
elif bat_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0, 0, 0, 1, 0, 0, 0, 0]
elif bat_team == 'Mumbai Indians':
    temp_array = temp_array + [0, 0, 0, 0, 1, 0, 0, 0]
elif bat_team == 'Rajasthan Royals':
    temp_array = temp_array + [0, 0, 0, 0, 0, 1, 0, 0]
elif bat_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0, 0, 0, 0, 0, 0, 1, 0]
elif bat_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0, 0, 0, 0, 0, 0, 0, 1]

bowl_team = input('Enter name of Bowling Team:')
if bowl_team == 'Chennai Super Kings':
    temp_array = temp_array + [1,0,0,0,0,0,0,0]
elif bowl_team == 'Delhi Daredevils':
    temp_array = temp_array + [0,1,0,0,0,0,0,0]
elif bowl_team == 'Kings XI Punjab':
    temp_array = temp_array + [0,0,1,0,0,0,0,0]
elif bowl_team == 'Kolkata Knight Riders':
    temp_array = temp_array + [0,0,0,1,0,0,0,0]
elif bowl_team == 'Mumbai Indians':
    temp_array = temp_array + [0,0,0,0,1,0,0,0]
elif bowl_team == 'Rajasthan Royals':
    temp_array = temp_array + [0,0,0,0,0,1,0,0]
elif bowl_team == 'Royal Challengers Bangalore':
    temp_array = temp_array + [0,0,0,0,0,0,1,0]
elif bowl_team == 'Sunrisers Hyderabad':
    temp_array = temp_array + [0,0,0,0,0,0,0,1]


inn = int(input('Enter innings:'))
bal = float(input('Enter ball:'))
run_w_bat = int(input('ENter runs with bat:'))
ex = int(input('Enter extras:'))
wid = int(input('Enter wides:'))
nob = int(input('Enter no balls:'))
by = int(input('Enter byes:'))
lby = int(input('Enter legbyes:'))
pen = int(input('Enter penalty:'))
ru = run_w_bat + ex + wid + nob + by + lby + pen
tot_runs = total

temp_array = temp_array + [inn, bal, run_w_bat, ex , wid,nob, by, lby, pen, ru, tot_runs]
data = np.array([temp_array])
my_prediction = int(regressor.predict(data)[0])
lower_limit = my_prediction-10
upper_limit = my_prediction+5
print("The final predicted score:", my_prediction)
