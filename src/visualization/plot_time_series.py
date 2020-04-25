

import matplotlib.pyplot as plt
import seaborn as sns

# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# ...  baseline plots
# ... -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-


def plot_time_series(df, x, y) :

    sns.set_style("darkgrid")

    plt.figure()
    s = sns.lineplot(x = x, y = y, data = df, palette = 'Set1')
    plt.show()


