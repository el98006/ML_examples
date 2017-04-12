# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import KFold
from scipy import linalg, dot
import os
import matplotlib.pyplot as plt
import sklearn.metrics
import time
import random
from sklearn.metrics import precision_score
'''
Category:
A: Attributes, site address
C: Cases,  each user is assigned an unique case ID.  
V: Votes/Ratings, each 'C' record followed by one or more V records. each V record contains id of a 'A' record visited. 

['A'] contains list of all the sites
['C'] contains all the case ids for each user
['V'] contains the site id voted.
  
'''

def process_user_activity_fast(raw_data):
    ''' get data into lists, then create DataFrame from lists.
    Huge performance improvement over creating an empty DataFrame and append row based on  
    .loc operations. 
     
    12s vs 9m
    
    '''    
    rating =[]    
    for _, values in raw_data.iterrows():
        
        if values['category'] == 'C':
            user_id = values['value']
            continue
        if values['category'] == 'V':
            site_id = values['value']
            rating.append((user_id, site_id ))
    
    rating_df = pd.DataFrame(rating, columns=['userid','webid'])
    
    return rating_df
            
def prepare_user_activity_slow(user_activity):
    ''' this has serious performance issue
    '''
    
    tmp = 0
    nextrow = False
    lastindex = user_activity.index[len(user_activity)-1]    
    
    for index, _ in user_activity.iterrows():
        if(index <= lastindex ):
            if(user_activity.loc[index,'category'] == "C"):
                # append two columns userid and webid to the ra
                
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
    
    return user_activity

def load_data():
    dir_name = os.path.dirname(__file__)

    f_path = os.path.join(dir_name, "anonymous-msweb.data")
    try:
        csv_data = pd.read_csv(f_path, header=None, skiprows=7, dtype={3:unicode, 4:unicode})
    except OSError as e:
        print 'Error opening file {}, error:{}'.format(f_path, e)
        exit

    # exclude all the A records, the site records,
    user_activity = csv_data.loc[csv_data[0] != "A"]
    
    user_activity.columns = ['category','value','vote','desc','url']
    
    #extract only category and value columns  in k/v format
    #if key = 'C', value = case id,  key = 'V', value = webid.
    user_activity = user_activity[['category','value']]
    
    
    site_count = len(user_activity.loc[user_activity['category'] =="V"].value.unique())
    case_count = len(user_activity.loc[user_activity['category'] =="C"].value.unique())
    
    print ' user count: {},  site count: {}'.format(case_count, site_count)
    
    #creating item profile
    items = csv_data.loc[csv_data[0] == "A"]
    items.columns = ['record','webid','vote','desc','url']
    items = items[['webid','desc']]
    items['webid'].unique().shape[0]

    return user_activity, items

def get_rating_matrix(user_activity):
    
    print ('started processing raw data')
    time_tic = time.clock()
    
    #user_activity is a pandas DataFrame, column:userid, webid
    user_activity = process_user_activity_fast(user_activity)
    
    time_tok = time.clock()
    
    print 'data processing completed, time taken {}'.format(time_tok - time_tic)
    
    user_activity_sort = user_activity.sort_values(by='webid', ascending=True)
    
    user_activity['userid'].unique().shape[0]
    user_activity['webid'].unique().shape[0]
    
    num_webid = len(user_activity_sort['webid'])
    
    #add a rating column, default value: 1
    user_activity_sort['rating'] = pd.Series(np.ones((num_webid,)), index=user_activity.index)
    
    #create a pivot, index is userid, columns are different webid, value = count the occurrence of [userid, webid], set to 0 if none. 
    rating_matrix = user_activity_sort.pivot(index='userid', columns='webid', values='rating').fillna(0)
    
        
    return rating_matrix

def visualize_rating(df):
    '''visualize the ratings by a userid, retrieve rating data of userid=1
    '''
    print 'rating matrix columns: {}'.format(df.columns.values)
   
    #apply function pd.value_counts to each column, and get counts of unique values, indexed by unique value: 0, not rated; 1: rated, columns are webids. 
    count_rating_by_site = df.apply(pd.value_counts)
    
    #get row where index value = 1.0, the value is total number of rating. 
    df_count_rating = count_rating_by_site.ix[1.0]
    df_count_norating = count_rating_by_site.ix[0.0]
    
    total_num_rating = df_count_rating.sum()
    total_num_norating = df_count_norating.sum()
    print 'number of missing rating: {:,}, number of ratings: {:,}'.format(total_num_norating, total_num_rating)
    
    print 'Sparsity of rating matrix is {:.3f} %'.format(float(total_num_rating) *100 /(total_num_rating + total_num_norating))
    
    
    plot1 = df_count_rating.plot(kind='hist', log=True, bins=100, title='web sites by number of visits')
    plot1.set_xlabel('webid')
    plot1.set_ylabel('Num of visit (logrithmics')
    plt.show()
    
    #show the top 10 rated webid
    d2 = df_count_rating.sort_values(ascending=False)
    plot2 = d2[:10].plot(kind='bar', title='Top 10 visited sites')
    plot2.set_xlabel('webid')    
   
    #have to call plt.show to show the plot objects create by plot()
    plt.show()
 
def analyze_model(model, tfidf_matrix): 
    
    term_list = model.get_feature_names()
    
    num_items, _ = tfidf_matrix.shape
    print 'number of features: {}'.format(len(term_list))
    print 'number of items: {}'.format(num_items)
    
    weights = np.asarray(tfidf_matrix.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': term_list, 'weight': weights})
    top_terms =  weights_df.sort_values(by='weight', ascending=False).head(20)
    
    print "top 20 term:\n"
    print top_terms.to_string(index=False)

    
    item_profile = tfidf_matrix.toarray()
    #item_profile: item projected into topic vector space, described by topics
    
    #for a given item, list the features with weight > 0. get a random item 
    for r in  np.arange(5):
        item_index = random.randint(0, num_items)
        topic_index = item_profile[item_index] > 0  
        item_topics = list(np.array(term_list)[topic_index])
        print '{}th item , the topic list is{}'.format(item_index, item_topics)
    
def split_data(x):
    '''
    fold = 0
    for train_idx, test_idx in kf.split(x):
        fold += 1 
        train_data = x[train_idx]
        test_data = x[test_idx]
        m = model(....)
        m.fit(train_data)
        score
    '''
    

        
def content_based_prediction():
    
    try:
        user_activity, items_raw = load_data()
    except:
        print "load data error"
        exit(0)
        
    #user_activity: columns 'category','value', key: 'V', 'C'; value: caseId, webid. 
    #item_raw: columns 'webid','desc'
   
    #rating: matrix : indexed by userid,  columns are webids: webid-0, webid-1, ... webid-n,
    rating = get_rating_matrix(user_activity)
    
    visualize_rating(rating)
    
    #items is a subset of items which appear in user_acitity, being rated, then sort 
    items = items_raw[items_raw['webid'].isin(rating.columns.tolist())]
    items = items.sort_values(by='webid', ascending=True)  
   
    # create v as  vectorizer, project items to vector space using tfidf based on 'desc', number of features:100
    v = TfidfVectorizer(stop_words ="english",max_features = 100,ngram_range= (0,3),sublinear_tf =True)
    
 
    #transform items desc to doc-term matrix, fit items into vectorizer, result: each row is a doc/webid, columns are 100 features. value is TF-IDF value
    tfidf_matrix = v.fit_transform(items['desc'])
    
    analyze_model(v, tfidf_matrix)
    
    
    item_profile = tfidf_matrix.toarray()
    
    # dot product of rating matrix and item profile to get user_profile, user are projected to the same tfidf feature vector space 
    user_profile = dot(rating, item_profile) / ( linalg.norm(rating) * linalg.norm(item_profile) )
    
    
    # recommendations based on the similarity between  user profile an item profile
   
    similarityCalc = sklearn.metrics.pairwise.cosine_similarity(user_profile, item_profile, dense_output=True)
    
    final_pred= np.where(similarityCalc>0.6, 1, 0)

    return final_pred,items

if __name__ == '__main__':
    
    prediction, p_items = content_based_prediction()
    
    ''' 
    
    
    p_userids = user_ids.tolist()
    t_users = test_users.tolist()

  
    while cnt < 10:
        print 'userid: {}'.format(test_users[cnt])
        if test_users[cnt] in user_ids:
            
            # user_ids is a Int64Index object, user_ids.values is a numpy.ndarray
            p_index = p_userids.index(test_users[cnt])
            p_rating = prediction[p_index]
            t_rating = test_rating[cnt]
            print precision_score(t_rating, p_rating)
            
            cnt +=1
    '''
