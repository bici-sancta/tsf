

import sys, os
import numpy as np
import pandas as pd

from base_models import Primitive
from base_models import Mean
from plot_time_series import plot_time_series

import matplotlib.pyplot as plt
import seaborn as sns

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  set some directory locations
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

home_dir = "/home/mcdevitt/PycharmProjects/eoy_prj"
data_dir = "./data/"

raw_file = "./raw/sp500.csv"

os.chdir(home_dir)
os.chdir(data_dir)

ls_ticker = ['^VIX', '^DJI', '^GSPC'
             'GESLX',
             'SIVIX',
             'FLCPX',  # fidelity us large cap
             'FSMDX',  # fidelity mid cap index
             'TIP',  # iShares TIPS bond etf
             'AGG']  # iShares aggregate bond etf

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  read in some data sets
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

df_sp5c = pd.read_csv(raw_file, parse_dates=['Date'])

df_sp5c = df_sp5c.sort_values(by = 'Date')
df_sp5c['year'] = pd.DatetimeIndex(df_sp5c['Date']).year
df_sp5c['log_close'] = np.log10(df_sp5c['Close'])

df_sp5c.columns = df_sp5c.columns.str.lower()

alpha = pd.Timestamp('2020-01-01')
omega = pd.Timestamp('2020-01-31')
period = 'D'

X = df_sp5c[['date', 'close']]

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  baseline plots
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


plot_time_series(df_sp5c, 'date', 'log_close')

subset = df_sp5c[df_sp5c['year'] > 1979]
plot_time_series(subset, 'date', 'log_close')

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... naive prediction
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

last_known = Primitive()
target_date = pd.Timestamp('2017-03-29')

last_known.fit(X, target_date)
last_known_value = last_known.value_

alpha = pd.Timestamp('2020-01-01')
omega = pd.Timestamp('2020-01-31')
predict_date = pd.date_range(alpha, omega, freq = period)

last_known_predict = last_known.predict(np.array(predict_date))

X_hat = pd.DataFrame({'predict_date' : predict_date, 'value' : last_known_predict})

last_known_sse = last_known.sse()

print(X_hat)
print(round(last_known_sse))
#last_known.print_metrics()

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ... mean value prediction
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

moyen = Mean()

train_alpha = pd.Timestamp('2019-01-01')
train_omega = pd.Timestamp('2019-12-31')

X_train = X[X['date'] >= train_alpha]
X_train = X_train[X_train['date'] <= train_omega]

moyen.fit(X_train)
moyen_value = moyen.value_

alpha = pd.Timestamp('2020-01-01')
omega = pd.Timestamp('2020-01-31')
predict_date = pd.date_range(alpha, omega, freq = period)

moyen_predict = moyen.predict(np.array(predict_date))

X_hat = pd.DataFrame({'predict_date' : predict_date, 'value' : moyen_predict})

moyen_sse = moyen.sse()

print(X_hat)
print(round(moyen_sse))
#last_known.print_metrics()
