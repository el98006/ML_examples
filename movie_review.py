'''
Created on Apr 4, 2017

@author: eli
'''
import os
import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_files
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

vectorizer_list = [CountVectorizer(), TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)]
#traditional term count, tfidf
#tfidf sublinear_tf=True to use logarithmic TF.

classifier_list = [SVC(), SVC(kernel='linear'), MultinomialNB()]
classifier_names = ['SVC', 'SVC linear kernel', 'MultinomialNB',]
def evaluation(prediction, labels):
    
    return True


def load_data():
    '''Note
    total number of samples = 2000, split into train set(0.7) and test set (0.3), 1400 samples in train, 600 in test
    '''
    
    cur_dir = os.path.dirname(__file__)
    data_dir = os.path.join(cur_dir, 'movie_data/txt_sentoken/')
    data = load_files(data_dir)
    return train_test_split(data.data, data.target, test_size=0.2, random_state=96)    

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
    
    '''
    i = 450
    j = 10
    words = vectorizer.get_feature_names()[i:i+10]
    chunk = pandas.DataFrame(test_features[j:j+7,i:i+10].todense(), columns=words)
    print 'peek into test_features: {}'.format(chunk)
    '''
    return train_features, test_features

def classify(train_features, train_labels,test_features, test_labels, classifier):
    
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
    train_data, test_data, train_labels, test_labels = load_data()
    for classifier in classifier_list:
        for vectorizer in vectorizer_list:    
            print 'vectorizer:{}, classifier:{}'.format(vectorizer.__class__.__name__, classifier_names[classifier_list.index(classifier)]) 
            train_features, test_features = feature_extraction(train_data, test_data, vectorizer)
            prediction = classify(train_features, train_labels, test_features, test_labels, classifier)
            print classification_report(test_labels, prediction)
                
def evaluate_models_with_pipeline():
    train_data, test_data, train_labels, test_labels = load_data()
    model = Pipeline([('vectorizer', vectorizer_list), ('classifier', classifier_list)])
    model.fit(train_data, test_data, train_labels, test_labels)
    return model
        
if __name__ == '__main__':
    evaluate_models()
    