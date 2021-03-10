import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import svm
import time, datetime
import matplotlib.pyplot as plt

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
