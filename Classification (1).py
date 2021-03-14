#!/usr/bin/env python
# coding: utf-8

# # Classification of Posts into Relevant Channels 

# In[2]:


import gensim
from gensim.models import word2vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import pickle


#import itertools
from gensim.models.word2vec import Text8Corpus
#from glove import Corpus, Glove


# # Loading the trained word2vec model
# 
# #### word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True,limit=100000)
# 
# #### After dumping into a pickle file: model_nlp, we directly operate over it
# 

# In[3]:


with open('model_nlp','rb') as f:
    word_vectors=pickle.load(f)


# # Cleaning File using Regex

# In[4]:


text_linked="""Ready to be inspired?Applications for the 2021 Generation Google Scholarship:
for women in computer science (APAC) are now open
Learn more about eligibility and how you can apply at http://goo.gle/3kidA0l
Generation Google Scholarship: for women in computer science (formerly known as 
Women Techmakers Scholarship) aims to inspire and support students in the fields of computing and technology and become active role models and leaders in the industry. As a scholar, you will receive a cash award for the 2021-2022 academic year and be invited to attend the annual (virtual) Scholars' Retreat to connect with fellow scholars, network with Googlers and participate in a number of developmental workshops to help enhance your skills to prepare you for a better tomorrow.
Still on the fence?
Google is hosting an Info Session over Youtube Livestream on Monday, 15th March 2021 where you will get to meet former scholars and ask live questions! Sign up now at http://goo.gle/3uqUkCu
Want to learn more about the essay section? Check out my Blog on Medium :) - https://lnkd.in/eC4nU-x
Application closing on Monday, 29th March 2021."""

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
stopwords = stopwords.words('english')
stopwords.append('\n')

def get_processed_text(text):
    """
    input: text -> an entire string of text
    output: tokens -> a list containing all filtered words
    """
    tags = re.compile(r'<.*?>')
    tags.sub('', text)                                 # to remove content in HTML tags
    text = re.sub(r'http\S+', ' ', text)               # to remove URLs
    text = re.sub(r'[^\w\s]',' ', text)                 # to remove punctuations
    text = re.sub(r'[^a-zA-Z]', ' ', text).lower()     # to remove anything other than characters
    tokens = [w for w in w_tokenizer.tokenize(text) if w not in stopwords and w[0] != '@'] # tokenizing across whitsepaces to extract words
    return tokens


# In[33]:


# The hashtags
categories=['women','job','coding','competition','scholarships','mentors','commerce','law','arts','creative','digital','session','event','finance']
match_of_categories=[]


# In[34]:


new_dataset=get_processed_text(text_linked)

for i in new_dataset:
    i=i.lower()
    


# In[35]:


def similar_one_out(words):
    """Accepts a list of words and returns the odd word"""
    match_of_categories.clear()
   
    for x in categories: 
        
        same_word=0
        
        try:
            avg_vector = word_vectors[x]
            
        except:
            for w in words:
                if(w==x):
                    match_of_categories.append(1)
                    same_word=1
                    
            if(same_word==1):
                continue
                
                
        #Iterate over every word and find similarity
        
        min_similarity = 0.5 #Very high value
        
        flag=0;
        
        words = new_dataset
        
        first_10=[]
        first_10.clear()

        for w in words:  
            try:
                temp=word_vectors[w]
            except:
                continue

            sim = cosine_similarity([temp],[avg_vector])

            sim=float(sim)
            
            if sim>=1.0:
                match_of_categories.append(1)
                flag=1
                break
            
            if sim>=0.5:
                t=[w,sim]
                first_10.append(t)
                
        if len(first_10)>=2 and flag==0:
            match_of_categories.append(1)
            
        elif len(first_10)<1 and flag==0:
            match_of_categories.append(0)
            
        first_10.clear()
            
    return match_of_categories


# In[36]:


copy=similar_one_out(new_dataset)


# In[37]:


print(categories)
print(copy)


# In[ ]:





# In[ ]:




