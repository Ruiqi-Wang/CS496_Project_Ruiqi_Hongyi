# Predicting stock movement with sentiment analysis of financial news

## Introduction
This is a final project of Hongyi Guo and Ruiqi Wang in course COMP_SCI-496: Advanced topics in Deep learning. The main reference of this project is: 'Topic Modeling based Sentiment Analysis on Social Media
for Stock Market Prediction' by Nguyen and Shirai.

Building an accurate stock price movement prediction model is still a challenging problem. A very common type of feature for prediction is the historical prices. In addition, it is easy to notice that the mood of society toward a company can greately affect its stock price. To this end, our goal in this project is to forcast the movement of stock price with features of historical price movements and the news of the companies.

## Model
There are two main parts in our model. Firstly, we employed sentiment analysis model to extract the sentiment features from texts. Secondly, we input features of historical prices and sentiments to predictive models.

### Sentiment analysis model
The main contribution of [Nguyen and Shirai] is to propose a novel topic based sentiment analysis model, called Topic Sentiment Latent Dirichlet Allocation (TSLDA) method. The proposed model, TSLDA, which is an extended model of Latent Dirichlet Allocation (LDA), infers the topics and their sentiments simultaneously. In TSLDA, words in a sentence are classified into three categories: topic words, opinion words, and others. For different topics, which are also represented by word distributions, will have different opinion word distributions. For each topic, the distribution of opinion words should be different for different sentiments.
A graphical illustration of TSLDA model can be shown in the follows:

![TSLDA](https://github.com/Ruiqi-Wang/CS496_Project_Ruiqi_Hongyi/blob/main/src/1.jpg)

Where the notations given in the graph are given as follows:

![Notation](https://github.com/Ruiqi-Wang/CS496_Project_Ruiqi_Hongyi/blob/main/src/4.jpg)

A generation process in TSLDA is as follows:
1. Choose a distribution of background words
2. For each topic k:
  * Choose a distribution of topic words
  * For each sentiment s of topic k: Choose a distribution of sentiment words 
3. For each document d:
  * Choose a topic distribution
  * Choose a sentiment distribution
  * For each sentence m：
    - Choose a topic assignment
    - Choose a sentiment assignment
    - For each word in the sentence, Choose a word in Equation 1:
![Eq1](https://github.com/Ruiqi-Wang/CS496_Project_Ruiqi_Hongyi/blob/main/src/5.jpg)

By Collapsed Gibbs Sampling method, the topic and sentiment assignment of sentence m and document d can be sampled by the following equation:

![Eq2](https://github.com/Ruiqi-Wang/CS496_Project_Ruiqi_Hongyi/blob/main/src/2.jpg)

The topic and its sentiment in each sentence are gotten from the topic assignment and sentiment assignment in TSLDA. If there is a sentence expressing the sentiment j on a topic i we represent the tuple (i,j)=1, and 0 otherwise. The proportion of (i,j) over all sentences are calculated for each message. On transaction day t, the features are the average of the proportions over all messages on date t, for all topics and all sentiments.

### Predictive model
In the paper [Nguyen and Shirai], the authors recommended to use Supportive vector machine to predict the movement of stock price. In particular, the prediction problem is a classification problems of two classes (1 for up and 0 for down). The paper suggested that to predict the movement on day t, the features should be decided to be tslda weights on day t and t-1, and the historical movements on day t-1 and t-2.

Alternatively, we also proposed to use recurrent neural network for prediction, since the input features are sequential data. We tested the performance of simple RNN and LSTM.

## Implementation
### Dataset
Our text data are financial news in between 2020.1.1 and 2021.1.1 on different companies. In addition, our dataset also includes the stock prices of these companies. Note that we only concern on the movements, these stock prices are transformed into 1/0 binary data representing going up or down. During the one year of 2020, we chose 10 stocks shown in the following chart. The column 'number of price' means the number of price movement data. The column 'number of news' means the number of days on which the company has a financial news. We split the data with 80% of training set and 20% of test set.

![Dataset](https://github.com/Ruiqi-Wang/CS496_Project_Ruiqi_Hongyi/blob/main/src/6.jpg)

### Implementing the TSLDA Gibbs Sampling

The implementation includes two parts, as the model does. In this section we will introduce our codes to extract the tslda weights from texts. 

#### Getting prepared with the data
The codes in this section can be found in 'src/Codes/Gibbs_Sampling/tslda_data.py'. We import the packages needed. Note that lemmatizer of sentences are needed. We employed sentiment words from 'Sentiwordnet 3.0: An enhanced lexical resource for sentiment analysis and opinion mining.'
```
import re
import json
from pandas import read_csv
from datetime import datetime
from collections import defaultdict
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
opinion_words = set(read_csv(
    'data/SentiWordNet_3.0.0.txt', comment='#', sep='\t', header=None)[4].tolist())
```

We defined the class of texts, in particular, the words, sentences and documents.
```
class Document:
    def __init__(self, raw):
        self.sentences = [Sentence(sent.strip('.')) for sent in nltk.sent_tokenize(raw)]


class Sentence:
    def __init__(self, sentence):
        # remove punctuations
        sentence = re.sub(r'[^\w\s.]', '', sentence)
        # tokenize
        self.words = [Word(word) for word in nltk.word_tokenize(sentence)]
        self.wordset = set()
        self.wordcalc = [defaultdict(lambda: 0) for _ in range(3)]
        self.topic = None
        self.sentiment = None
        # categorization
        for i, word in enumerate(self.words):
            if word.part == 'n' and i > 0 and self.words[i - 1].part == 'n':
                # consecutive nouns
                word.category = 1
                self.words[i - 1].category = 1
            elif word.lemma in opinion_words:
                word.category = 2
            else:
                word.category = 0
        # construct word sets
        for word in self.words:
            self.wordset.add(word.lemma)  # for V_{d,m}
            self.wordcalc[word.category][word.lemma] += 1  # for W^{*,*}_{d,m,v,c}


class Word:
    def __init__(self, word):
        self.lemma = lemmatizer.lemmatize(word.lower())
        synsets = wordnet.synsets(self.lemma)
        self.part = synsets[0].pos() if len(synsets) > 0 else None
        self.category = None
```

A TSLDAData class is defined for preprocessing the data:

```
class TSLDAData:
    def __init__(self, stock):
        self._prices = read_csv(f'data/prices/{stock}_prices.csv')
        self._messages = json.load(open(f'data/news/{stock}_news.json'))
        self.opinion_words = read_csv('data/SentiWordNet_3.0.0.txt', comment='#', sep='\t', header=None)
        self.all_messages = defaultdict(lambda: list())
        self.messages = list()
        self.prices = list()
        self.dates = list()
        self.preprocess()

    def preprocess(self):
        for idx in self._messages['date'].keys():
            date = datetime.strptime(self._messages['date'][idx], '%Y/%m/%d')
            if date_is_selected(date):
                self.all_messages[date].append(Document(self._messages['text'][idx]))

        self._prices = self._prices.sort_index(ascending=False)  # in the order of time
        last_day = None
        for i, p in self._prices.iterrows():
            if not last_day:
                last_day = p['price']
            else:
                date = datetime.fromisoformat(p['date'])
                adj_close = p['price']  # the adjusted close prices
                if date in self.all_messages.keys():
                    self.prices.append(int(adj_close < last_day))
                    self.messages.append(self.all_messages[date])
                    self.dates.append(date)
                last_day = adj_close

    def __call__(self):
        return self.messages, self.prices, self.dates
```
Since we are not supposed to provide the raw data we used, if you want to employ the codes on your own news and price data, please save your news data in dictionary with file name '{Stock Name}\_news.json'. Your dictionary should be in format {{'date':yyyymmdd, 'news': '...'}, ...}. Please save your price data in csv file with file name '{Stock Name}\_prices.csv' where the first column is date and second column is price.
