import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as Data

from model_thick import thickLSTM
from thick_data import para as thick_paramter


def getRoolingAve(raw_data, rooling_window):
    out = []
    raw_data = np.array(raw_data)
    length = raw_data.shape[0]
    for i in range(length):
        temp_ave = raw_data[i:i + rooling_window].mean()
        out.append(temp_ave)
    return np.array(out)


def test_model(model, test_loader):
    loss = nn.MSELoss()
    loss_sum = 0
    result_list = []
    for step, (x, y) in enumerate(test_loader):
        x, y = x.to(torch.float32).to(device), y.to(torch.float32).to(device)
        _, _, res = model(x.unsqueeze(0).to(torch.float32))
        l = loss(res.squeeze(), y)
        loss_sum += l
        res = torch.flatten(res.detach())
        result_list.extend(np.array(res))
    print('loss:', loss_sum)
    return np.array(result_list)


def getTrainData(raw_train_feat, member_num, future_window_num):
    length = raw_train_feat.shape[0] - member_num - 1
    train_input = []
    train_label = []
    for i in range(length):
        temp_data = raw_train_feat[i:i + member_num]
        # temp_ave = raw_train_feat[i + member_num:i + member_num + future_window_num].mean()
        # temp_std = raw_train_feat[i + member_num:i + member_num + future_window_num].std()

        # 连续预测
        # label = raw_train_feat[i + member_num:i + member_num + future_window_num]
        # 单点预测
        label = raw_train_feat[i + member_num]
        train_input.append(temp_data)
        train_label.append(label)
    return torch.tensor(np.array(train_input).astype(np.float64)), torch.tensor(np.array(train_label).astype(np.float64))


def train_thick_model(model, train_loader):
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    max_epoch = 3000

    for epoch in range(max_epoch):
        loss_sum = 0.0
        print('Starting Epoch:', epoch)

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(torch.float32).to(device), y.to(torch.float32).to(device)
            batch_size = len(x)
            optimizer.zero_grad()
            _, _, res = model(x.unsqueeze(0).to(torch.float32))
            l = loss(res.squeeze(), y)
            loss_sum += l
            l.backward()
            optimizer.step()
        print('epoch:', epoch, 'loss:', loss_sum / batch_size)


def getTimeGapThickData(origin_data, start_date, end_date):
    slice_thick_data = origin_data[np.where(origin_data[:, -1] > start_date)]
    slice_thick_data = slice_thick_data[np.where(slice_thick_data[:, -1] < end_date)]
    return slice_thick_data


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
history_window = 80
future_window = 30
if __name__ == '__main__':
    data_path_root = './thick_data'
    # thick_cols = thick_paramter.cols
    thick_raw = pd.read_csv(data_path_root + r'/厚度在线采集0222-0226.csv', names=[chr(i) for i in range(97, 116)])
    # b目标值 q实时值 r周期值 s时间
    thick_data = thick_raw[['b', 'q', 'r', 's']]
    thick_data.sort_values('s', inplace=True)

    thick_data_np = np.array(thick_data.loc[:, ['b', 'r', 's']])
    # 获取时间间隔中的数据
    start_date = '2023-02-22 21:00:00.000'
    end_date = '2023-02-22 23:00:00.000'
    thick_data_np = getTimeGapThickData(thick_data_np, start_date, end_date)
    thick_data_np = thick_data_np[:, 1] - thick_data_np[:, 0]

    # plt.plot(range(3200), thick_data_np[1800:5000])
    input_feat = thick_data_np[1800:5000]
    test_feat = thick_data_np[6000:9000]

    # z-score
    input_feat = (input_feat - input_feat.mean()) / input_feat.std()
    test_feat = (test_feat - test_feat.mean()) / test_feat.std()

    [train_input, train_label] = getTrainData(input_feat, history_window, future_window)
    [test_input, test_label] = getTrainData(test_feat, history_window, future_window)

    # train_data
    batch_size = 50
    train_dataset = Data.TensorDataset(train_input, train_label)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True)

    # test_data
    test_dataset = Data.TensorDataset(test_input, test_label)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=False)

    model = thickLSTM(history_window_length=history_window)

    model = model.to(device)
    train_thick_model(model, train_loader)
    torch.save(model, '0312_lstm_thick.pth')

    test_out = test_model(model, test_loader)

    test_length = test_out.shape[0]

    label_ave = getRoolingAve(test_label[0:test_length], future_window)
    out_ave = getRoolingAve(test_out, future_window)

    plt.figure(figsize=(100, 6.0))
    plt.subplot(3, 1, 1)
    plt.title(str(history_window) + '-' + str(future_window))
    plt.plot(range(test_length), test_out, color='r', label='model')
    plt.plot(range(test_length), test_label[0:test_length], color='b', label='origin')
    plt.subplot(3, 1, 2)
    plt.title('rooling average')
    plt.plot(range(test_length), out_ave[0:test_length], label='origin')
    plt.plot(range(test_length), label_ave[0:test_length], label='origin')
    plt.axhline(0)
    plt.subplot(3, 1, 3)
    plt.plot(range(test_length), test_feat[0:test_length], color='g', label='weight')
    plt.show()
