#Install all the libraries the project needs
#pip install -r requirements.txt 

import nltk
import re
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords') #collection of stopwords
from sklearn.model_selection import train_test_split
from nltk.classify import NaiveBayesClassifier

import warnings #only for os
warnings.filterwarnings("ignore") #only for os

#Loading data
nltk.download('twitter_samples') #collection of tweets 

#View list of files in twitter_samples
file_list = twitter_samples.fileids()

print("List of files in twitter_samples:")
for file_id in file_list:
    print(file_id)

#Variables creation
pos_tweets = twitter_samples.strings('positive_tweets.json')
neg_tweets = twitter_samples.strings('negative_tweets.json')

#Tokenization = breaks down a text into individual words or tokens
tokenizer = TweetTokenizer(preserve_case = False, strip_handles = True, reduce_len = True)

#Cleansing text from useless characters and removes them from the original text

#obtains a list of stopwords in English
stop_words = stopwords.words('english')

#creates an instance of PorterStemmer, which enables text stemming using Porter's algorithm.
#stemming words = reducing them to their root or basic form, removing any suffixes or prefixes
stemmer = PorterStemmer()

def clean_tweet(tweet):
    #removes all dollar signs followed by a sequence of alphanumeric characters from the tweet
    tweet = re.sub(r'\$\w*', '', tweet)  
    #revomes all URL links in the tweet
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)  
    #removes all '#' in the tweet
    tweet = re.sub(r'#', '', tweet)  
    #tokenize tweet = divide text into individual words or tokens
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        #checks if the word is not a stopword and if it only consists of alphabetic characters.
        if (word not in stop_words and word.isalpha()):
            #applies stemming to each tokenized word in the tweet
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    #returns the list of cleaned words 
    return tweets_clean

#Feature Extraction
#this def calculates the frequency of each word in a list of words and returns a dictionary containing:
#the words as keys and their frequencies as values.
def get_word_frequency(words):
    word_frequency = {}
    for word in words:
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1
    return word_frequency

#Cleansing the tweets
pos_cln_tokens = [clean_tweet(tweet) for tweet in pos_tweets]
neg_cln_tokens = [clean_tweet(tweet) for tweet in neg_tweets]    

#Calculates the word frequencies
pos_feat = [(get_word_frequency(tokens), 'Positive') for tokens in pos_cln_tokens]
neg_feat = [(get_word_frequency(tokens), 'Negative') for tokens in neg_cln_tokens] 
all_feat = pos_feat + neg_feat

#Verification of data balance

#Label's count 
num_pos = sum(1 for _, label in all_feat if label == 'Positive')
num_neg = sum(1 for _, label in all_feat if label == 'Negative')

#View distribution
print('Number of positive examples:', num_pos)
print('Number of negative examples:', num_neg)

#Calculation of proportions
total = len(all_feat)
pos_ratio = round((num_pos / total) * 100)
neg_ratio = round((num_neg / total) * 100)

print('\nPositive ratio:', pos_ratio,'%')
print('Negative ratio:', neg_ratio,'%')

if pos_ratio >= 0.5:
    print ('\nThe dataset is balanced')
else:
    print ('\nThe dataset is not balanced')


#Slitting and label examples with train for 80% and test for 20%
train_set, test_set = train_test_split(all_feat, test_size = 0.2, random_state = 42)

#Trains a Naive Bayes classifier using the 'train_set' training set
classifier = NaiveBayesClassifier.train(train_set)

#Test the model and calculates the accuracy
def get_accuracy(test_set, classifier):
    correct = 0
    for (features, label) in test_set:
        prediction = classifier.classify(features)
        if prediction == label:
            correct += 1
    accuracy = float(correct) / len(test_set)
    return accuracy

accuracy = get_accuracy(test_set, classifier)
print('\nThe Accuracy is:', (accuracy * 100),'%')

#Test the model on the new data
def predict_sentiment(tweet):
    cleaned_tweet = clean_tweet(tweet)
    features = get_word_frequency(cleaned_tweet)
    return classifier.classify(features)

#Test examples:
print('\nTest:')
#Example of right positive result
tweet1 = 'This movie was fantastic!I would like to watch it again'
print('\nThe tweet -> "',tweet1,'" is ', predict_sentiment(tweet1))

#Example of WRONG negative result
tweet2 = "This chicken was nasty!"
print('\nThe tweet -> "',tweet2,'" is', predict_sentiment(tweet2))

#Example of right result
tweet3 = "I'm in love the dogs"
print('\nThe tweet -> "',tweet3,'" is', predict_sentiment(tweet3),'--> WRONG! Should be positive!')

#Example of right result
tweet4 = "I'm a neutral comment"
print('\nThe tweet -> "',tweet4,'" is', predict_sentiment(tweet4))

#Example of WRONG negative result
tweet5 = "This pic is so bad!"
print('\nThe tweet -> "',tweet5,'" is', predict_sentiment(tweet5))