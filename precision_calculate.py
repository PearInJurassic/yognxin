#!/home/panchenyu/anaconda3/bin/python3.9

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import random
import torch.utils.data as Data
import csv
from utils import *
import matplotlib.pyplot as plt


# 长时间的预测函数
# 用于离线预测未来的一段数据，递进式预测
def predict_test(model, first_set):
    result_list = list(first_set)
    round = future_window
    x = torch.tensor(first_set)
    for i in range(round):
        x = x.to(torch.float32).to(device)
        _, _, res = model(x.unsqueeze(0).unsqueeze(0).to(torch.float32))
        res = torch.flatten(res.detach())
        result_list.extend(np.array(res))
        x = torch.tensor(result_list[-history_window:])
    return np.array(result_list[history_window:])


# 离线的数据测试
def exam_offline_predict(path, model):
    global aim_weight
    raw_data = pd.read_csv(path)
    raw_data.sort_values('Logtime', inplace=True)
    weight_data = np.array(raw_data.loc[:, machine_weight_name])

    precision_list = []
    model_predict_ave = []
    true_ave = []
    aim_weight_line = []
    adjust_record = []
    sum_of_abs_predict = 0
    sum_of_abs_original = 0

    # 采前7500个数据作为测试
    data_begin_idx = 12000
    data_end_idx = 18080

    # ----------- 预测逻辑部分开始，可以不用理解 ------------#
    for i in range(data_begin_idx, data_end_idx):
        aim_weight_line.append(aim_weight)
        # 模拟到达数据
        print('No.', i, 'data has arrived')
        raw_data = np.array(weight_data[data_begin_idx-history_window:i + future_window])
        aim_weight = float(aim_weight)
        raw_data -= aim_weight
        offset_mean = raw_data.mean()
        offset_std = raw_data.std()

        input_set = (raw_data[-(history_window + future_window):-future_window] - offset_mean) / offset_std

        out_list = predict_test(model, input_set) * offset_std + offset_mean

        out_list = np.concatenate((out_list, input_set[-history_window // 2:]))
        out_ave = out_list.mean()
        model_predict_ave.append(out_ave)
        true_ave.append(raw_data[-(future_window + history_window // 2):].mean())
        # ----------- 预测逻辑部分结束 ------------#

        # out_ave:模型输出的[t-40,t+30]的均值
        # true_ave:实际情况下的[t-40,t+30]的均值
        # 准确率计算
        precision_list.append(1 - (abs(out_ave - true_ave[-1]) / (true_ave[-1] + aim_weight)))

        # 显示实际目标值的变化情况
        print(aim_weight, out_ave + aim_weight, true_ave[-1] + aim_weight)
        print(raw_data.mean())
        print('set mean:', 1 - (abs(out_ave - true_ave[-1]) / (true_ave[-1] + aim_weight)))

        # 计算预测和原始的绝对值偏差量
        sum_of_abs_predict += abs(out_ave)
        sum_of_abs_original += abs(raw_data[-(future_window + history_window // 2):].mean())

        # TODO 可以做一个观察周期
        # 数据实际0.3s来一个，可以设置一个周期进行
        if abs(out_ave) > 1.0:
            # 调整值通常以0.05为最小单位
            suggest_adjust = int(-out_ave / 0.8) * 0.05
            adjust_record.append(suggest_adjust)
        else:
            adjust_record.append(0.0)
        # 模拟学习率使调整目标值梯度回归到实际均值
        if abs(offset_mean) < 0.7 * 2.0:
            aim_weight += offset_mean * 0.005

    # 绘图的数据范围可以自己调整，但不要超过前面的data_begin_idx和data_end_idx
    plt_data_begin_idx = max(80, data_begin_idx)
    plt_data_end_idx = max(7500, data_end_idx)
    plt.figure(figsize=(100, 6.0))

    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(range(plt_data_end_idx - plt_data_begin_idx), model_predict_ave, 'b', label='predict')
    ax1.plot(range(plt_data_end_idx - plt_data_begin_idx), true_ave, 'g', label='true')
    ax1.legend(loc='upper left')

    # 双y轴
    ax2 = ax1.twinx()
    ax2.plot(range(plt_data_end_idx - plt_data_begin_idx), adjust_record, 'r', label='model advice')
    ax2.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(range(plt_data_end_idx - plt_data_begin_idx), aim_weight_line)

    plt.show()

    precision_list = np.array(precision_list)
    print('-------------------------\n')
    print('Offline examination data num:', plt_data_end_idx - plt_data_begin_idx)
    print('Sum of predict offset:', sum_of_abs_predict)
    print('Sum of original offset:', sum_of_abs_original)
    print('Average precision:', precision_list.mean())

    true_ave = np.array(true_ave)
    model_predict_ave = np.array(model_predict_ave)
    print('Average offset loss:', (true_ave - model_predict_ave).mean())


# 历史窗口为80
history_window = 80
future_window = 30
# 获取算力设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
machine_weight_name = 'Gage1_smooth'

if __name__ == '__main__':
    # file_path = r'./质量检测0210/NDC_202302221630.csv'
    file_path = r'/Users/chenyupan/永信/3.27 调整数据/NDC_202303271328.csv'
    # aim_weight = 35.0
    aim_weight = 37.0

    model_path = '11.22_lstm_76.0_0.3.pth'
    # 载入模型
    model = torch.load(model_path, map_location=device)
    model.to(device)
    exam_offline_predict(file_path, model)
