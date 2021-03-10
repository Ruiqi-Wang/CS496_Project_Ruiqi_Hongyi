# Predicting stock movement with sentiment analysis of financial news

## Introduction
This is a final project of Hongyi Guo and Ruiqi Wang in course COMP_SCI-496: Advanced topics in Deep learning. The main reference of this project is: 'Topic Modeling based Sentiment Analysis on Social Media
for Stock Market Prediction' by Nguyen and Shirai.

Building an accurate stock price movement prediction model is still a challenging problem. A very common type of feature for prediction is the historical prices. In addition, it is easy to notice that the mood of society toward a company can greately affect its stock price. To this end, our goal in this project is to forcast the movement of stock price with features of historical price movements and the news of the companies.

## Model
There are two main parts in our model. Firstly, we employed sentiment analysis model to extract the sentiment features from texts. Secondly, we input features of historical prices and sentiments to predictive models.

### Sentiment analysis model
The main contribution of [Nguyen and shirai] is to propose a novel topic based sentiment analysis model, called Topic Sentiment Latent Dirichlet Allocation (TSLDA) method. 
