
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_excel('data/Eclipse_4sourcev1.xls', header=None, names=['severity', 'summary', 'description'], usecols=[0, 1, 2])
df


# In[4]:


severity = df['severity']
counts = severity.value_counts(ascending=True)

total_count = len(severity)

class_probability = [0] * 6


for i in range(1, 6):
    class_probability[i] = counts.loc[i] / total_count
    


if sum(class_probability) != 1:
    raise Error("Probability mismatch.")


# In[17]:


class_group = df.groupby('severity')
class_group.get_group(1)['summary']

