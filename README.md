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
Our text data are financial news in between 2020.1.1 and 2021.1.1 on different companies. In addition, our dataset also includes the stock prices of these companies. Note that we only concern on the movements, these stock prices are transformed into 1/0 binary data representing going up or down. During the one year of 2020, we chose 10 stocks shown in the following chart. The column 'number of price' means the number of price movement data. The column 'number of news' means the number of days on which the company has a financial news. 

![Dataset](https://github.com/Ruiqi-Wang/CS496_Project_Ruiqi_Hongyi/blob/main/src/6.jpg)


## Implementation
