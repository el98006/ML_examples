# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 22:36:10 2016

@author: Suresh
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import linalg, dot
import os
import matplotlib.pyplot as plt
import sklearn.metrics

'''
Category:
A: Attributes, site address
C: Cases,  user activities, 
V: the values of Cases, each 'C' record followed by one or more V records. each V record contains id of a 'A' record visited. 

['A'] contains list of all the sites
['C'] contains all the cases /users
  
'''

def content_based():
    
    dir_name = os.path.dirname(__file__)
    
    f_path = os.path.join(dir_name, "anonymous-msweb.data")
    raw_data = pd.read_csv(f_path,header=None,skiprows=7)

    #creating user profile
    user_activity = raw_data.loc[raw_data[0] != "A"]
    
    user_activity.columns = ['category','value','vote','desc','url']
    
    #extract only first two columns
    user_activity = user_activity[['category','value']]
    
    
    site_count = len(user_activity.loc[user_activity['category'] =="V"].value.unique())
    case_count = len(user_activity.loc[user_activity['category'] =="C"].value.unique())
    
    print ' case/rating count: {},  site_count: {}'.format(case_count, site_count)

    tmp = 0
    nextrow = False
 
    
    lastindex = user_activity.index[len(user_activity)-1]

    for index, row in user_activity.iterrows():
        if(index <= lastindex ):
            if(user_activity.loc[index,'category'] == "C"):
                # append two columns userid and webid to the user_activity dataframe
                tmp = 0
                
                userid = user_activity.loc[index,'value']
                tmp = userid
                nextrow = True            
                # C records always followed by V records, 
            elif(user_activity.loc[index,'category'] == "V" and nextrow == True):
               
                    webid = user_activity.loc[index,'value']
                    user_activity.loc[index,'webid'] = webid
                    # retrieve userid from previous C record, temporarily stored in tmp
                    user_activity.loc[index,'userid'] = tmp
                    if(index != lastindex and user_activity.loc[index+1,'category'] == "C"):
                        # the last 'V' record for previous C record
                        nextrow = False
                    
    # only keep all V records, which contains both webids and userids                       
    user_activity = user_activity[user_activity['category'] == "V" ]
    
    # only keep columns userid and webid
    user_activity = user_activity[['userid','webid']]
    user_activity_sort = user_activity.sort('webid', ascending=True)
    
    user_activity['userid'].unique().shape[0]
    user_activity['webid'].unique().shape[0]
    
    plt.hist(user_activity[webid])
    plt.show()                        
    
    
    sLength = len(user_activity_sort['webid'])
    
    #add a rating column, default value: 1
    user_activity_sort['rating'] = pd.Series(np.ones((sLength,)), index=user_activity.index)
    
    #create a pivot, index is userid, columns are different webid, value = count the occurence of [userid, webid], set to 0 if none. 
    rating_matrix = user_activity_sort.pivot(index='userid', columns='webid', values='rating').fillna(0)
    
    rating_matrix = rating_matrix.to_dense().as_matrix()
    
    #creating item profile
    items = raw_data.loc[raw_data[0] == "A"]
    items.columns = ['record','webid','vote','desc','url']
    items = items[['webid','desc']]
    items['webid'].unique().shape[0]
    
    items_rated = items[items['webid'].isin(user_activity['webid'].tolist())]
    items_rated_sorted = items_rated.sort('webid', ascending=True)
    
    #project items to vector space using tfidf based on 'desc'
    
    v = TfidfVectorizer(stop_words ="english",max_features = 100,ngram_range= (0,3),sublinear_tf =True)
    
    v.get_feature_names()
    
    #transform items desc to doc-term matrix 
    x = v.fit_transform(items_rated_sorted['desc'])
    
    item_profile = x.to_dense()
    np.savetxt("tf_idf",x.todense())
    
    # dot product of rating matrix and item profile to get user_profile 
    
    user_profile = dot(rating_matrix,item_profile)/linalg.norm(rating_matrix)/linalg.norm(item_profile)
    
    
    # recommendations based on the smilarity between  user profile an item profile
 
    
    similarityCalc = sklearn.metrics.pairwise.cosine_similarity(user_profile, item_profile, dense_output=True)
    
    
    final_pred= np.where(similarityCalc>0.6, 1, 0)
    np.savetxt('pred', final_pred )
    final_pred.shape
    
   
    
if __name__ == '__main__':
    content_based()
    