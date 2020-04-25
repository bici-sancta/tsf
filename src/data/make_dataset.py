# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...

import os
import datetime
import pandas as pd
import numpy as np
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

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  read in some data sets
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

sp5c = pd.read_csv(raw_file, parse_dates=['Date'])

sp5c = sp5c.sort_values(by = 'Date')
sp5c['year'] = pd.DatetimeIndex(sp5c['Date']).year
sp5c['log_close'] = np.log10(sp5c['Close'])

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  baseline plots
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

sns.set_style("darkgrid")

plt.figure()
s = sns.lineplot(x = 'Date', y = 'log_close', data = sp5c, palette = 'Set1')

subset = sp5c[sp5c['year'] > 1979]
plt.figure()
s = sns.lineplot(x = 'Date', y = 'log_close', data = subset, palette = 'Set1')

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  create lag column
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

sp5c['lag_log_close'] = sp5c.log_close.shift(periods = 1)

plt.figure()
s = sns.scatterplot(x = 'lag_log_close', y = 'log_close', data = sp5c,
                    palette = 'Set1',
                    hue = 'year',
                    alpha = 0.5)


subset = sp5c[sp5c['year'] > 2017]
plt.figure()
s = sns.scatterplot(x = 'lag_log_close', y = 'log_close', data = subset,
                palette = 'Set1',
                hue = 'year',
                alpha = 0.3,
                legend = "full")



# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  centered moving average
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

moving_average_length = 50

sp5c['moving_average'] = sp5c.log_close.rolling(moving_average_length).mean()

sp5c['lagged_moving_average'] = sp5c.moving_average.shift(periods = int(-moving_average_length/2))

plt.figure()
s = sns.lineplot(x = 'Date', y = 'log_close', data = sp5c, palette = 'Set1')
s = sns.lineplot(x = 'Date', y = 'lagged_moving_average', data = sp5c, palette = 'Set1')

subset = sp5c[sp5c['year'] > 1999]
subset = subset[subset['year'] < 2006]
plt.figure()
s = sns.lineplot(x = 'Date', y = 'log_close', data = subset, palette = 'Set1')
s = sns.lineplot(x = 'Date', y = 'lagged_moving_average', data = subset, palette = 'Set1')

subset = sp5c[sp5c['year'] == 2018]
plt.figure()
s = sns.lineplot(x = 'Date', y = 'log_close', data = subset, palette = 'Set1')
s = sns.lineplot(x = 'Date', y = 'lagged_moving_average', data = subset, palette = 'Set1')

