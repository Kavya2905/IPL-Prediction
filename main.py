import sys
import pandas as pd
import numpy as np
import joblib

from predictor import predictRuns

score = predictRuns('inputFile.csv')
print('Predicted Score:', score, 'runs.')

