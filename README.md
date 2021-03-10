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

