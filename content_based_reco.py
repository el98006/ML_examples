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

SIMILARITY_THRESHHOLD = 0.7
 
''' 
Content-based recommenders exploit solely ratings provided by the active user to build her own profile. In contrast, Content-based recommenders 
exploit solely ratings provided by the active user to build her own profile.

To construct and update USER PROFILE of an active user, her reactions to items are collected and stored in FEEDBACK/ANNOTATION. 
PROFILE LEARNER applies learning algorithm to generate a predictive model - user profile -, which will be used by FILTERING COMPONENT.
given a new item representation, FILTERING COMPONENT predicts whether it will be liked or disliked by by the active user by comparing 
features in the item representation - item profile - to those in the representation of user preferences -- user profile.  

drawback:
- Serendipity
- New User 

'''

def process_user_feedback_fast(raw_data):
    '''
    Category:
    A: Attributes, site address
    C: Cases,  each user is assigned an unique case ID.  
    V: Votes/Ratings, each 'C' record followed by one or more V records. each V record contains id of a 'A' record visited. 
    
    ['A'] contains list of all the sites
    ['C'] contains all the case ids for each user
    ['V'] contains the site id voted.
      

    get data into lists, then create DataFrame from lists.
    Huge performance improvement over creating an empty DataFrame and append row based on  
    .loc operations. 
     
    12s vs 9m
    
    '''    
    feedback =[]    
    for _, values in raw_data.iterrows():
        
        if values['category'] == 'C':
            user_id = values['value']
            continue
        if values['category'] == 'V':
            site_id = values['value']
            feedback.append((user_id, site_id ))
    
    feedback_df = pd.DataFrame(feedback, columns=['userid','webid'])
    
    return feedback_df
            
def prepare_user_feedback_slow(user_activity):
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

def load_data(train=True):
    
    
    if train == True:
        filename = 'anonymous-msweb.data'
    else:
        filename = 'anonymous-msweb.test'
    
    dir_name = os.path.dirname(__file__)

    f_path = os.path.join(dir_name, filename)
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
    
    print ' user count: {},  webid count: {}'.format(case_count, site_count)
    
    #creating item profile
    items = csv_data.loc[csv_data[0] == "A"]
    items.columns = ['record','webid','vote','desc','url']
    items = items[['webid','desc']]
    items['webid'].unique().shape[0]

    return user_activity, items

def get_rating_matrix(user_feedback):
    '''learning users's interests based on user likes or dislikes a training document set. 
    '''
    print ('started processing raw data')
    time_tic = time.clock()
    
    #user_feedback is a pandas DataFrame, column:userid, webid
    user_feedback = process_user_feedback_fast(user_feedback)
    
    time_tok = time.clock()
    
    print 'data processing completed, time taken {}'.format(time_tok - time_tic)
    
    user_activity_sort = user_feedback.sort_values(by='webid', ascending=True)
    
    user_feedback['userid'].unique().shape[0]
    user_feedback['webid'].unique().shape[0]
    
    num_webid = len(user_activity_sort['webid'])
    
    #add a rating column, default value: 1
    user_activity_sort['rating'] = pd.Series(np.ones((num_webid,)), index=user_feedback.index)
    
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
    #item_profile: items projected into topic/term vector space, described by topics
    
    #for a given item, list the features with weight > 0. draw 5 random items
    
    ''' 
    for r in  np.arange(5):
        item_index = random.randint(0, num_items)
        topic_index = item_profile[item_index] > 0  
        item_topics = list(np.array(term_list)[topic_index])
        print '{}th item , the topic list is{}'.format(item_index, item_topics)
    '''

    

        
def content_based_prediction(user_activity, items_raw):
    #return a prediction dataframe, indexed by userid, columns
    
    try:
        user_activity, items_raw = load_data(train=True)
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

        
    # create v as a vectorizer, generate item profiles using tfidf based on 'desc', number of features:100
    v = TfidfVectorizer(stop_words ="english",max_features = 100,ngram_range= (0,3),sublinear_tf =True)
    
 
    #transform items desc to doc-term matrix, fit items into vectorizer, result is item profile: each row is a doc/webid, columns are 100 features. value is TF-IDF value
    tfidf_matrix = v.fit_transform(items['desc'])
    
    analyze_model(v, tfidf_matrix)
    
    
    item_profile = tfidf_matrix.toarray()
    
    # user profile is generated based on their feedbacks on items, and the item profiles. Their favorite items 
    # dot product of rating matrix and item profile to get user_profile, user are projected to the same tfidf feature vector space  
    user_profile = dot(rating, item_profile) / ( linalg.norm(rating) * linalg.norm(item_profile) )
    
    
    # recommendations based on the similarity between  user profile an item profile
    similarityCalc = sklearn.metrics.pairwise.cosine_similarity(user_profile, item_profile, dense_output=True)
    
    # return elements, if condition is true, return 1 otherwise 0.
    final_pred= np.where(similarityCalc > SIMILARITY_THRESHHOLD, 1, 0)
    
    df_prediction = pd.DataFrame(final_pred, index=rating.index, columns=rating.columns)

    return df_prediction

def items_not_in_training_data(train_items_raw, test_item):
    
    train_set = set(train_items_raw['webid'].unique())
    test_set = set(test_item['webid'].unique())
    
    diff_items = [u for u in test_set if u not in train_set]
    print ' number of webids in training set:{}, test set:{}'.format(len(train_set), len(test_set))
    if len(diff_items):
        print 'different items in training set and test set: {}'.format (diff_items)
        return diff_items
    else: 
        return False


def validate_test_training(test, training):
    scores = []
    rec_idx = 0 
    for rec_idx in range(len(test)):
        '''
        userids = test.index
        rec_idx = random.randrange(len(userids)) 
        '''
        #select row by row number, either should work
        #target = test[:rec_idx]
        target = test.iloc[rec_idx]
        uid = target.name
        
        #print "pick a random record from the test set, {}th record, uid={}".format(rec_idx, uid)
        #print "start validation\n-------------------------" 
        #print "result:"
        #print target[target > 0].index.values
         
        
        training_rec = training.ix[uid]
        #print " prediction:"
        #print training_rec[training_rec > 0].index.values
        #print "----------------------- new instance---------"
        
        hit = 0
        miss = 0
        
        actual_values = set(target[target > 0].index.values)
        predicted_values = set(training_rec[training_rec > 0].index.values)

        for i in actual_values:
            if i in predicted_values:
                hit += 1
            else:
                miss +=1
                
        # hit rate/recall = TP/P = TP((TP+FN), miss rate = FN/P
        if hit + miss > 0:     
            hit_rate = hit / (hit + miss)
        else:
            hit_rate = 0
        scores.append(hit_rate)
        
        rec_idx += 1         

    print ("average hit rate {:.2f}%").format(float(np.array(scores).sum())*100/len(scores))
            
            
    
if __name__ == '__main__':
    
    try:
        train_user_activity, train_items_raw = load_data(train=True)
        test_activity, test_item = load_data(train=False)
    except:
        print "load data error"
        exit(0)
    
    prediction = content_based_prediction(train_user_activity, train_items_raw)

    new_items = items_not_in_training_data(train_items_raw, test_item)
    if new_items:    
        print 'following items don\'t exist in training data {}'.format(new_items)
        exit(1)
        
    test_rating = get_rating_matrix(test_activity)
    
    validate_test_training(test_rating, prediction)
    
    
   


