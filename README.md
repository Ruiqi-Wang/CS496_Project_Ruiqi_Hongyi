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
Since we are not supposed to provide the raw data we used, if you want to employ the codes on your own news and price data, please save your news data in dictionary with file name '{Stock Name}\_news.json'. Your dictionary should be in format {{'date':yyyymmdd, 'news': '...'}, ...}. Please save your price data in csv file with file name '{Stock Name}\_prices.csv' where the first column is date and second column is price. You should keep the files in 'Gibbs_sampling/data' folder.

#### Implementation of Gibbs Sampling based on TSLDA
To implement Gibbs_sampling, we defined TSLDA class. Function calc_num_words() counts the total number of words in each document. Function gibbs_sampling() helps the TSLDA class to sample over the given data, following the TSLDA rule. Function pretrain(T) operates gibbs_sampling() for T times. If TSLDA is called, it returns the desired weight features. The codes can be found in 'src/Codes/Gibbs_sampling/tslda.py'.
```
class TSLDA:
    def __init__(self, documents, alpha, beta, gamma, lam, T, K, S=3):
        self.documents = documents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lam = lam
        self.n_topics = K
        self.n_sentiments = S
        self.counter = [
            [[defaultdict(lambda: 0) for _ in range(3)]
             for _ in range(self.n_sentiments)]
            for _ in range(self.n_topics)]  # shape is (K x S x 3), for W^{a,b}_{*,*,v,c}
        self.group_by_topic = np.zeros(self.n_topics)
        self.group_by_sentiment = np.zeros(self.n_sentiments)
        self.n_words = self.calc_num_words()
        self.pretrain(T)

    def calc_num_words(self):
        wordset = set()
        for doc in self.documents:
            for sent in doc.sentences:
                wordset.update(sent.wordset)
        return len(wordset)

    def pretrain(self, T):
        for _ in trange(T):
            self.gibbs_sampling()

    def gibbs_sampling(self):
        for doc in self.documents:
            for sent in doc.sentences:
                # sent.topic, sent.sentiment = multinomial()
                p = np.zeros((self.n_topics, self.n_sentiments))
                for a in range(self.n_topics):
                    for b in range(self.n_sentiments):
                        Za = self.group_by_topic[a] - int(sent.topic == a)
                        Zb = self.group_by_sentiment[b] - int(sent.sentiment == b)
                        num1 = np.prod([
                            np.prod([
                                sum(self.counter[a][b_][1][key] for b_ in range(self.n_sentiments))
                                - int(sent.topic == a) * sent.wordcalc[1][key]
                                + self.alpha + j
                                for j in range(sent.wordcalc[1][key])
                            ]) for key in sent.wordset])
                        den1 = np.prod([
                            sum(sum(self.counter[a][b_][1].values()) for b_ in range(self.n_sentiments))
                            - int(sent.topic == a) * sum(sent.wordcalc[1].values())
                            + self.n_words * (self.alpha + j)
                            for j in range(sum(sent.wordcalc[1].values()))])
                        num2 = np.prod([
                            np.prod([
                                self.counter[a][b][2][key]
                                - int(sent.topic == a and sent.sentiment == b) * sent.wordcalc[2][key]
                                + self.lam + j
                                for j in range(sent.wordcalc[2][key])
                            ]) for key in sent.wordset])
                        den2 = np.prod([
                            sum(self.counter[a][b][2].values())
                            + self.n_words * (self.lam + j)
                            for j in range(sum(sent.wordcalc[2].values()))])
                        p[a, b] = (Za + self.beta) * (Zb + self.gamma) * num1 / den1 * num2 / den2
                if sent.topic is not None:
                    self.group_by_topic[sent.topic] -= 1
                    self.group_by_sentiment[sent.sentiment] -= 1
                    for word in sent.words:
                        self.counter[sent.topic][sent.sentiment][word.category][word.lemma] -= 1
                p += 1e-7
                p /= p.sum()
                ind = np.random.multinomial(1, p.flatten()).argmax()
                sent.topic = ind // self.n_sentiments
                sent.sentiment = ind % self.n_sentiments
                self.group_by_topic[sent.topic] += 1
                self.group_by_sentiment[sent.sentiment] += 1
                for word in sent.words:
                    self.counter[sent.topic][sent.sentiment][word.category][word.lemma] += 1

    def __call__(self, documents):
        weights = np.zeros((self.n_topics, self.n_sentiments))
        for doc in documents:
            for sent in doc.sentences:
                weights[sent.topic, sent.sentiment] += 1
        weights /= weights.sum()
        return weights.flatten()
```

To obtain the tslda features, run src/Codes/Gibbs_sampling/main.py:

```
import argparse
import numpy as np
import pandas as pd
from tslda import TSLDA
from tslda_data import TSLDAData

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--K', type=int, default=10, help='# of topics.')
    parser.add_argument('-s', '--S', type=int, default=3, help='# of sentiments')
    parser.add_argument('-a', '--alpha', type=float, default=0.1, help='Dirichlet prior vectors.')
    parser.add_argument('-b', '--beta', type=float, default=0.01, help='Dirichlet prior vectors.')
    parser.add_argument('-g', '--gamma', type=float, default=0.01, help='Dirichlet prior vectors.')
    parser.add_argument('-l', '--lam', type=float, default=0.1, help='Dirichlet prior vectors.')
    parser.add_argument('-p', '--test-proportion', type=float, default=0.2, help='Proportion of test set')
    parser.add_argument('-t', '--T', type=int, default=10, help='Iterations for Gibbs sampling')
    parser.add_argument('--stock', type=str, default='xom', help='Stock name.')
    return parser.parse_args()
```

parse_args() are defined to set the model parameters. By default, we set the TSLDA model to have 10 topics and each topic has 3 sentiments. Alpha, beta, gamma and lambda are prior parameters, whose default values are suggested in [Nguyen and Shirai]. To run gibbs samplings, run main():

```
def main():
    args = parse_args()
    messages, prices, dates = TSLDAData(args.stock.upper())()
    documents = [doc for msg in messages for doc in msg]
    tslda = TSLDA(documents, args.alpha, args.beta, args.gamma, args.lam, args.T, args.K, args.S)
    pd.DataFrame([[date] + tslda(news).tolist() for news, date in zip(messages, dates)]).to_csv(
        f'data/news/{args.stock.upper()}_news.csv', index=False)

if __name__ == '__main__':
    main()

```

### Implementing predictive model
The codes for this part are in the folder 'src/Codes/predict'. We have obtained the TSLDA weight from running Gibbs sampling codes. The weight are saved in 'src/Codes/predict/news' in csv files '{Stock Name}\_news.csv'. Here we try svm, RNN and LSTM model for prediction. First load needed packages:
```
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import svm
import time, datetime
import matplotlib.pyplot as plt
```
Load the data. If you have obtained the tslda data, please save them in the same format. **Note that the TSLDA data is not available every day, since there are days on which there is no news. On those days without news, set the tslda feature to be 1/(K\*S), meaning that the news does not provide any information.**

```
def preload(stock_name, start_date, end_date):
    # stock_name: name of stock to be predict
    # start_date, end_date: time range of predictive model. datetime.date, format '%Y-%M-%D'

    price_filename = 'prices/' + stock_name + '_prices.csv'
    tslda_filename = 'news/' + stock_name + '_news.csv'
    prices = pd.read_csv(price_filename)
    tslda = pd.read_csv(tslda_filename)

    # Date transformation:
    t_date = []
    for i in range(len(prices)):
        date_str = prices['date'][i]
        y, m, d = time.strptime(date_str, '%Y-%m-%d')[:3]
        date = datetime.date(y, m, d)
        t_date.append(date)
    prices['t_date'] = t_date
    prices.drop('date', axis=1, inplace=True)

    t_date = []
    for i in range(len(tslda)):
        date_str = tslda['0'][i]
        y, m, d = time.strptime(date_str, '%Y-%m-%d')[:3]
        date = datetime.date(y, m, d)
        t_date.append(date)
    tslda['t_date'] = t_date
    tslda.drop('0', axis=1, inplace=True)

    movements = []
    dates = []
    features = []
    N = 30
    count_message_date = 0
    for i in range(len(prices)):
        if end_date >= prices['t_date'][i] >= start_date:
            date = prices['t_date'][i]
            dates.append(date)
            movements.append(int(prices['price'][i] > prices['price'][i - 1]))
            if date in tslda['t_date'].to_list():
                p = tslda['t_date'].to_list().index(date)
                features.append(np.asarray(tslda.iloc[p, 0:-1], dtype='float64'))
                count_message_date = count_message_date + 1
            else:
                features.append(np.ones(N) / N)
    df = pd.DataFrame({'date': dates, 'feature': features, 'movement': movements})

    return df, count_message_date, len(df)
```
Predict with SVM model, features being news and prices in svm_predict() and price only for svm_predict_priceonly():
```
def svm_predict(df, n_train, n):
    clf = svm.SVC()
    X, y = [], []
    for t in range(2, n):
        price_feature = np.concatenate([np.eye(2)[df['movement'][t-1]], np.eye(2)[df['movement'][t-2]]])
        tslda_feature = np.concatenate([df['feature'][t], df['feature'][t-1]])
        X.append(np.concatenate([price_feature, tslda_feature]))
        y.append(df['movement'][t])

    clf.fit(X[:n_train], y[:n_train])
    print('svm accuracy:', np.mean(y[n_train:] == clf.predict(X[n_train:])))

def svm_predict_priceonly(df, n_train, n):
    clf = svm.SVC()
    X, y = [], []
    for t in range(2, n):
        price_feature = np.concatenate([np.eye(2)[df['movement'][t-1]], np.eye(2)[df['movement'][t-2]]])
        X.append(price_feature)
        y.append(df['movement'][t])

    clf.fit(X[:n_train], y[:n_train])
    print('svm price only accuracy:', np.mean(y[n_train:] == clf.predict(X[n_train:])))
```

Predict with RNN model, features being news and prices for rnn_predict() and price only for rnn_predict_priceonly() :
```
class rnn_predictor(nn.Module):
    def __init__(self, input_size, hidden_size, seq_len, batch_size=1):
        super(rnn_predictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        # self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.hidden2out = nn.Linear(seq_len * hidden_size, 1)

    def forward(self, input):
        # Size of input: batch_size x seq_len x input_size
        hidden, _ = self.RNN(input) # batch_size x seq_len x hidden_size
        hidden = torch.flatten(hidden, start_dim=1, end_dim=2) # batch_size x seq_len * hidden_size
        output = self.hidden2out(hidden) # batch_size x 1
        output = torch.sigmoid(output)  # batch_size x 1
        return output

    def predict(self, input, threshold=0.5):
        output = self.forward(input)
        return output.item() > threshold

def rnn_predict(df, n_train, n, seq_len=10, num_epoch=100, input_size=32, hidden_size=16):
    with torch.no_grad():
        rnn_dataset = torch.zeros(1, n, input_size)
    for t in range(1, n):
        feature_t = np.concatenate([np.eye(2)[df['movement'][t-1]], df['feature'][t]])
        rnn_dataset[0, t, :] = torch.from_numpy(feature_t)

    RNN = rnn_predictor(input_size, hidden_size, seq_len)
    optim = torch.optim.Adam(RNN.parameters(), lr=1e-3)
    losses = []
    losses_epoch = []
    for epoch in range(num_epoch):
        for t in range(seq_len + 1, n_train):
            optim.zero_grad()
            y = RNN.forward(rnn_dataset[:, (t - seq_len):t, :])
            loss = - (1 - df['movement'][t]) * torch.log(1 - y) - df['movement'][t] * torch.log(y)
            losses_epoch.append(loss)
            loss.backward()
            optim.step()
        losses.append(torch.mean(torch.Tensor(losses_epoch)))
        losses_epoch = []
    # plt.figure(figsize=(10, 5))
    # plt.title("Loss")
    # plt.plot(losses)
    # plt.show()
    accuracy = []
    for t in range(n_train, n):
        y_pred = RNN.predict(rnn_dataset[:, (t - seq_len):t, :])
        accuracy.append(y_pred == df['movement'][t])

    print('rnn accuracy:', np.mean(accuracy))


def rnn_predict_priceonly(df, n_train, n, seq_len=10, num_epoch=50, input_size=2, hidden_size=16):
    with torch.no_grad():
        rnn_dataset = torch.zeros(1, n, input_size)
    for t in range(1, n):
        feature_t = np.eye(2)[df['movement'][t-1]]
        rnn_dataset[0, t, :] = torch.from_numpy(feature_t)

    RNN = rnn_predictor(input_size, hidden_size, seq_len)
    optim = torch.optim.Adam(RNN.parameters(), lr=1e-3)
    losses = []
    losses_epoch = []
    for epoch in range(num_epoch):
        for t in range(seq_len + 1, n_train):
            optim.zero_grad()
            y = RNN.forward(rnn_dataset[:, (t - seq_len):t, :])
            loss = - (1 - df['movement'][t]) * torch.log(1 - y) - df['movement'][t] * torch.log(y)
            losses_epoch.append(loss)
            loss.backward()
            optim.step()
        losses.append(torch.mean(torch.Tensor(losses_epoch)))
        losses_epoch = []
    # plt.figure(figsize=(10, 5))
    # plt.title("Loss")
    # plt.plot(losses)
    # plt.show()
    accuracy = []
    for t in range(n_train, n):
        y_pred = RNN.predict(rnn_dataset[:, (t - seq_len):t, :])
        accuracy.append(y_pred == df['movement'][t])

    print('rnn price only accuracy:', np.mean(accuracy))
```
To run the tests on different stocks, run the following codes:
```
STOCK_NAME = ['GOOG', 'WMT', 'TSLA', 'IBM', 'EBAY', 'WBA', 'UNH', 'FCX', 'PEP', 'XRX']
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2021, 1, 1)
test_proportion = 0.2
for stock_name in STOCK_NAME:
    df, n_message, n = preload(stock_name, start_date, end_date)
    print('\nStock name: ', stock_name, '\tn_message: ', n_message, '\tn: ', n)
    n_train = n - int(n * test_proportion)
    svm_predict(df, n_train, n)
    svm_predict_priceonly(df, n_train, n)
    rnn_predict(df, n_train, n)
    rnn_predict_priceonly(df, n_train, n)
```
Output example:
```
Stock name:  GOOG 	n_message:  122 	n:  260
svm accuracy: 0.46
svm price only accuracy: 0.46
rnn accuracy: 0.5576923076923077
rnn price only accuracy: 0.5192307692307693
```
## Results
The prediction accuracy for different algorithms are given as follows:
![Dataset](https://github.com/Ruiqi-Wang/CS496_Project_Ruiqi_Hongyi/blob/main/src/7.jpg)
