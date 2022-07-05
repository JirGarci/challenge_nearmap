import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
path = "/Users/jairgarcia/Documents/psv_test/"

# %%cellcode DATA IMPORT AND VERIFICATION

df = pd.read_csv(path + 'test.psv', sep ="|", skiprows = 1)
df.head(4)
df['dates'] = pd.to_datetime(df['dates']) #CONVERTS TO TIME DATE DATA TYPE
df.info() #CHECK FOR NANs AND OTHER NUISANCES

# %%cellcode DATE OPERATIONS AND BASIC QUERYING
date_range = str(df['dates'].dt.date.min()) + ' to ' +str(df['dates'].dt.date.max())
date_range

#DATA RANGE
df['dates'].dt.date.max() - df['dates'].dt.date.min()
#OBTAIN WEEK'S DAY FROM DATE
df['days'] = pd.to_datetime(df['dates']).dt.day_name()
df.head(10)

# %%cellcode SUBSET FRAME TO EXTRACT DATA FROM THURSDAYS ONLY AND GET F1 METRIC
th = df[df['days']=='Thursday']
th.head(10)
f1 = f1_score(th['y'].values, th['yhat'].values)
print("F1 score for Thursdays is {0:.5F}".format(f1))


th.describe( ) #SUMMARY STATISTICS
th['res'] = (th['yhat']-th['y'])**2
#RMSE BETWEEN OBSERVED AND PREDICTED AS A SECONDARY MEASURE OF ACCURACY
np.sqrt(np.sum((th['yhat']-th['y'])**2/th.shape[0]))

#EOF
