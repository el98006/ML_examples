'''
Created on Apr 4, 2017

@author: eli
'''
import os
import pandas
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_files
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import time

from sklearn.ensemble.forest import RandomForestClassifier

vectorizer_list = [CountVectorizer(), TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)]
vectorizer_names = ['CountVectorizer', 'TfidfVectorizer']
#traditional term count, tfidf
#tfidf sublinear_tf=True to use logarithmic TF.

classifier_list = [SVC(), SVC(kernel='linear'), MultinomialNB(), RandomForestClassifier(n_estimators = 100)]
classifier_names = ['SVC', 'SVC linear kernel', 'MultinomialNB', 'RandomForest']

VERBOSE_MODE = False




def load_data():
    '''Note
    total number of samples = 2000, split into train set(0.7) and test set (0.3), 1400 samples in train, 600 in test
    '''
    
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, 'movie_data/txt_sentoken/')
    data = load_files(data_dir)
    train_data, test_data, train_labels, test_labels = train_test_split(data.data, data.target, test_size=0.2, random_state=96)
  
    return train_data, test_data, train_labels, test_labels  
        

def feature_extraction(train_data, test_data, vectorizer):
    '''Note
    Note: vectorizer is used to convert docs into feature space
    '''

    
    train_features = vectorizer.fit_transform(train_data)
    '''
    fit() build the dictionary from the raw docs. fit_transform() creates the vocabulary and feature weights 
    from the training data. 
    '''  
    

    test_features = vectorizer.transform(test_data)
    # transform test data using the same dictionary built inside vectorizer in previous step,
    # and return test_features,  cast test_data into test_feature space.     
    
    print 'num of docs, num of features in trainset : {}'.format(train_features.shape)
    print 'shape of test_features {}'.format(test_features.shape)
    
    if VERBOSE_MODE is True:
        i = 450
        j = 10
        words = vectorizer.get_feature_names()[i:i+10]
        chunk = pandas.DataFrame(test_features[j:j+7,i:i+10].todense(), columns=words)
        print 'peek into test_features: {}'.format(chunk)
    
    return train_features, test_features

def classify(train_features, train_labels,test_features, classifier):
    
    '''
    Note:
    train the classifier with train_features and train_lables, then predict test_labels based on test_features
    '''
    classifier.fit(train_features, train_labels)
    prediction = classifier.predict(test_features)
    return prediction
   
   
def evaluate_models():
    '''Note
    F1 score 2*((precision*recall)/(precision+recall)), F1 score conveys the balance between the precision and the recall
    '''
    scores={}
    
    train_data, test_data, train_labels, test_labels = load_data()
  
    for ci, classifier in enumerate(classifier_list):
        for vi, vectorizer in enumerate(vectorizer_list):    
            print 'vectorizer:{}, classifier:{}'.format(vectorizer.__class__.__name__, classifier_names[classifier_list.index(classifier)])
            t_start = time.time() 
            train_features, test_features = feature_extraction(train_data, test_data, vectorizer)
            prediction = classify(train_features, train_labels, test_features,  classifier)
            time_spent = time.time() - t_start
            scores[f1_score(test_labels, prediction)] = (ci, vi, time_spent)
            print classification_report(test_labels, prediction)
    sorted_scores = sorted(scores, reverse =True)
    ranking = [ (scores[r], r) for r in sorted_scores]
    print '{:<20} {:<20} {:<5} {:<5}'.format('classifier', 'Vectorizer', 'F1', 'Time')
    for (ci, vi, time_spent), f1 in ranking:
        print '{:<20} {:<20} {:<5.2f} {:>5.2f}s'.format(classifier_names[ci], vectorizer_names[vi], f1, time_spent)
                
def evaluate_models_with_pipeline():
    '''Note
    Pipeline two interfaces: transformer and estimator.
    GridSearchCV performs an exhaustive search on specified parameter values. 
    This allows iteration through defined sets of parameters and the reporting of the result in the form of various metrics.
     
    '''
  
    steps = [('tfidf', TfidfVectorizer()), ('classifier', SVC())]
    
    '''
    Note: for pipeline, param grid formats: prepend estimator name: tfidf__, or classifier__ to parameter name 
    to find the right syntax format, checkout the output of 'sorted(pipeline.get_params().keys())'
    total 3x3x2 = 18 combinations, GridSearchCV will iterate through 18 sets of params and calculate their scroes. 
    '''  
    params = {
              'tfidf__min_df': [5, 10, 20], 
              'tfidf__max_df': [0.8, 0.9, 0.95],
              'classifier__kernel': ['linear', 'rbf']
             }
        
    
    train_data, test_data, train_target, test_target = load_data()
    pipeline = Pipeline(steps)
    
    models = GridSearchCV(pipeline, cv=3, n_jobs=2, param_grid=params, refit=True)
    # cv=3 3 folds cross validation for each estimator to get the scores. 
    
    tic = time.time()
    print 'Start GridSearchCV against pipeline...'
    models.fit(train_data, train_target)
    mean_scores = np.array(models.cv_results_['mean_test_score'])
    tok = time.time()
    
    print 'Time taken for Grid Search {:.0f} seconds'.format(tok - tic)
    print 'mean score list: {}'.format(mean_scores)
    
    _,  best_params = sorted(zip(models.cv_results_['rank_test_score'], models.cv_results_['params']))[0] 
    print 'the best parameters for classifier: {}'.format(best_params)
    
        
if __name__ == '__main__':
    # evaluate different classifier performance with different vectorizer
    evaluate_models()
    #user GridSearchCV 
    evaluate_models_with_pipeline()