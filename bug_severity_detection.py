
# coding: utf-8

# In[93]:


import pandas as pd


# In[94]:


df = pd.read_excel('data/Eclipse_4sourcev1.xls', header=None, names=['severity', 'summary', 'description'], usecols=[0, 1, 2])
test_df = df.iloc[-700:, :]
df = df.iloc[:-700, :]

test_df


# In[95]:


severity = df['severity']
counts = severity.value_counts(ascending=True)

total_count = len(severity)

class_probability = [0] * 6


for i in range(1, 6):
    class_probability[i] = counts.loc[i] / total_count
    


if sum(class_probability) != 1:
    raise Error("Probability mismatch.")


# In[96]:


class_group = df.groupby('severity')

from collections import defaultdict
word_count_per_class = defaultdict(lambda : [0, 0, 0, 0, 0, 0])
total_words_per_class = [0, 0, 0, 0, 0, 0]

vocabulary = set()
for key, group in class_group:    
    summaries = group['summary']    
    for summary in summaries:        
        words = summary.split()        
        total_words_per_class[key] += len(words)
        for word in words:
            vocabulary.add(word)
            word_count_per_class[word][key] += 1
        
    
        

(len(vocabulary), total_words_per_class)



        


# In[97]:


def probability_of_word_given_class(words, clazz):
    "P(Word|Class) = (Occurence of word in class + 1) / (total words in class + total words in vocabulary)"
    
    denominator = total_words_per_class[clazz] + len(vocabulary)
    
    for word in words:
        yield (word_count_per_class[word][clazz] + 1) / denominator
        


# In[98]:


from functools import reduce
import operator

def predict_severity(summary):
    """
    P(Class|Words) = P(Word_1|Class) * P(Word_2|Class) ... * P(Class)
        
    class = argmax(P(Class|Words))
    """
    
    words = summary.split()
    class_probability_given_summary = [0, 0, 0, 0, 0, 0]
    
    for clazz in range(1,6):
        product_of_words_probability = (reduce(operator.mul, probability_of_word_given_class(words, clazz)))
        
        class_probability_given_summary[clazz] = product_of_words_probability  * class_probability[clazz]
    
    return class_probability_given_summary.index(max(class_probability_given_summary))



        


# In[99]:


from pprint import pprint

def measure_accuracy(test_data):
    result_matrix = [[0, 0, 0, 0, 0, 0] for _ in range(6)]
        
    for index, data in test_data.iterrows():
        prediction = predict_severity(data['summary'])
        real = data['severity']
        result_matrix[prediction][real] += 1
        
#     pprint([row[1:] for row in result_matrix[1:]])
    pprint(result_matrix)
    
    
    precisions = [0] * 6
    recalls = [0] * 6
    for clazz in range(1, 6):        
        precisions[clazz] = result_matrix[clazz][clazz] / sum(result_matrix[clazz])
        recalls[clazz] = result_matrix[clazz][clazz] / sum([row[clazz] for row in result_matrix])
    
    print(precisions)
    print(recalls)
    
    precision = sum(precisions) / (len(precisions) - 1)  #the array is 1 size larger
    recall = sum(recalls) / (len(recalls) - 1)  #the array is 1 size larger
    f_measure = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f_measure
    
    
    
    
    
measure_accuracy(test_df)

    

