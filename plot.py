#installation of Pandas and Seaborn required
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt

# read the data and convert some variables from seconds to hours
data = pd.read_csv('zaidhaque_jawbone_data.csv')
data.update(data.s_awake_time / 3600)
data.update(data.s_asleep_time / 3600)
data.update(data.s_duration / 3600)
data.update(data.m_inactive_time / 3600)
data.update(data.m_active_time / 3600)
data.update(data.m_lcit / 3600)

# line graph for sleeping and waking times
fig, ax = plt.subplots()
ax.plot(np.arange(len(data.DATE)), data.s_awake_time, 'o-', color='b', label='Waking')
ax.plot(np.arange(len(data.DATE)), data.s_asleep_time, 'o-', color='r', label='Sleeping')
legend = ax.legend(loc='upper right', shadow=True)
plt.xticks(range(0, len(data.DATE), 10), ['{0}/{1}'.format(str(date)[4:6], str(date)[6:]) for date in list(data.DATE.loc[range(0, len(data.DATE), 10)])], rotation=40)
plt.xlabel('Date')
plt.ylabel('Hours from midnight')
plt.title('Sleeping & Waking Times', fontsize=18)
plt.show()

# hexbin plot for steps walked and hours slept
sns.set(style="ticks")
ax = sns.jointplot(data.m_steps, data.s_duration, kind='hex', stat_func=kendalltau, color='#4CB391')
ax.set_axis_labels(xlabel='Steps walked', ylabel='Hours slept')
plt.show()

# bar chart for active and inactive time
active_t = np.array(data.m_active_time)
inactive_t = np.array(data.m_inactive_time) - np.array(data.s_duration)
active_per = active_t / (active_t + inactive_t) * 100
inactive_per = inactive_t / (active_t + inactive_t) * 100
plot_data = pd.DataFrame({'active': active_per, 'inactive': inactive_per})
ax = plot_data.plot(kind='bar', stacked=True)
labels = [item.get_text() for item in ax.get_xticklabels()]
for idx, label in enumerate(labels):
    if (idx + 1) % 5 == 0:
        labels[idx] = '{0}/{1}'.format(str(data.DATE.loc[idx])[4:6], str(data.DATE.loc[idx])[6:8])
    else:
        labels[idx] = ''
plt.setp(ax.get_xticklabels(), rotation=60)
plt.ylim((0, 100))
plt.xlabel('Date')
plt.ylabel('Percentage')
plt.title('Daily Active vs. Inactive time', fontsize=18)
ax.set_xticklabels(labels)
plt.show()

# line graph for 
fig, ax = plt.subplots()
ax.plot(np.arange(len(data.DATE)), data.m_inactive_time, 'o-', color='b', label='Total inactive time')
ax.plot(np.arange(len(data.DATE)), data.m_lcit, 'o-', color='r', label='Longest consecutive inactive time')
legend = ax.legend(loc='upper right', shadow=True)
plt.xticks(range(0, len(data.DATE), 10), ['{0}/{1}'.format(str(date)[4:6], str(date)[6:]) for date in list(data.DATE.loc[range(0, len(data.DATE), 10)])], rotation=40)
plt.xlabel('Date')
plt.ylabel('Inactive time (hours)')
plt.title('Total inactive time vs. Longest consecutive inactive time', fontsize=18)
plt.show()