#!/usr/bin/env python
# coding: utf-8

# **RandomForest_baseline.py**

# In[4]:


# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
import os

# 训练数据：src/step1/input/ctbu_train_data.txt
# 测试数据：src/step1/input/ctbu_test_sample.txt
# 结果文件：src/step1/ground_truth/test_prediction.csv

def getPrediction():
    #1 数据准备
    ############################
    #1.1 读入数据
    #1.1.1 读入训练数据
    train  = []
    ID     = []
    track  = []
    target = []
    label  = []
    ############################
    # with open('src/step1/input/ctbu_train_data.txt', 'r') as data:
    with open('/home/aistudio/work/ctbu_train_data.txt', 'r') as data:
        for i in data:
            k = i.split()
            train.append(i)
            ID.append(k[0])
            track.append(k[1])
            target.append(k[2])
            label.append(k[3])

    train_data = pd.DataFrame([ID, track, target, label]).T
    train_data.columns = ['id', 'track', 'target', 'label']
    # print(train_data.head())

    #1.1.2 读入测试数据
    test   = []
    ID     = []
    track  = []
    target = []
    ############################
    # with open('src/step1/input/ctbu_test_data.txt', 'r') as data:
    with open('/home/aistudio/work/ctbu_test_data.txt', 'r') as data:
        for i in data:
            k = i.split()
            test.append(i)
            ID.append(k[0])
            track.append(k[1])
            target.append(k[2])
    test_data = pd.DataFrame([ID, track, target]).T
    test_data.columns = ['id', 'track', 'target']
    # print(test_data.head())

    #2 数据探索
    # 略
    #3 特征工程
    #
    # # 假设机器轨迹与人类轨迹区别在于速度和斜率变化有不同规律
    #
    def v_feature(track):
        #
        # # 函数：计算轨迹速度
        #
        x = []
        y = []
        t = []
        k = track.rstrip(';').split(';')
        for i in k:
            j = i.split(',')
            x.append(float(j[0]))
            y.append(float(j[1]))
            t.append(float(j[2]))
        x1 = []
        y1 = []
        t1 = []
        for i in range(len(x) - 1):
            x1.append(x[i + 1] - x[i])
            y1.append(y[i + 1] - y[i])
            t1.append(t[i + 1] - t[i])
        x1 = np.array(x1)
        y1 = np.array(y1)
        t1 = np.array(t1)
        v = np.sqrt(x1 ** 2 + y1 ** 2) / (t1 + 10 ** -10)
        if len(v) == 0:
            v = np.array(0)
        return v
    ##
    def grid_feature(track):
        #
        # # 函数：计算轨迹斜率
        #
        x = []
        y = []
        k = track.rstrip(';').split(';')
        for i in k:
            j = i.split(',')
            x.append(float(j[0]))
            y.append(float(j[1]))
        x1 = []
        y1 = []
        for i in range(len(x) - 1):
            x1.append(x[i + 1] - x[i])
            y1.append(y[i + 1] - y[i])
        x1 = np.array(x1)
        y1 = np.array(y1)
        g = y1 / (x1 + 10 ** -10)
        if len(g) == 0:
            g = np.array(0)
        return g

    # 构造速度和斜率的统计基础特征
    train_data['v_max']  = train_data.track.apply(lambda x: v_feature(x).max())
    train_data['v_min']  = train_data.track.apply(lambda x: v_feature(x).min())
    train_data['v_mean'] = train_data.track.apply(lambda x: v_feature(x).mean())
    train_data['v_std']  = train_data.track.apply(lambda x: v_feature(x).std())
    train_data['g_max']  = train_data.track.apply(lambda x: grid_feature(x).max())
    train_data['g_min']  = train_data.track.apply(lambda x: grid_feature(x).min())
    train_data['g_mean'] = train_data.track.apply(lambda x: grid_feature(x).mean())
    train_data['g_std']  = train_data.track.apply(lambda x: grid_feature(x).std())

    test_data['v_max']   = test_data.track.apply(lambda x: v_feature(x).max())
    test_data['v_min']   = test_data.track.apply(lambda x: v_feature(x).min())
    test_data['v_mean']  = test_data.track.apply(lambda x: v_feature(x).mean())
    test_data['v_std']   = test_data.track.apply(lambda x: v_feature(x).std())
    test_data['g_max']   = test_data.track.apply(lambda x: grid_feature(x).max())
    test_data['g_min']   = test_data.track.apply(lambda x: grid_feature(x).min())
    test_data['g_mean']  = test_data.track.apply(lambda x: grid_feature(x).mean())
    test_data['g_std']   = test_data.track.apply(lambda x: grid_feature(x).std())

    def num(track):
        #
        # # 函数：统计轨迹点个数
        #
        x = []
        y = []
        k = track.rstrip(';').split(';')
        n = len(k)
        return n

    # 构造轨迹点数特征
    train_data['num'] = train_data.track.apply(lambda x: num(x))
    test_data['num']  = test_data.track.apply(lambda x: num(x))

    #4 数据划分
    #4.1 建立线下训练集和验证集
    X = train_data[['label', 'v_max', 'v_min', 'v_mean', 'v_std', 'g_max', 'g_min', 'g_mean', 'g_std', 'num']].copy()
    print(X)
    y = X['label']
 
    X.drop('label', axis=1, inplace=True)
    x_train, x_validation, y_train, y_validation = train_test_split(X, y)
    #4.2 建立预测数据集
    x_test = test_data[['v_max', 'v_min', 'v_mean', 'v_std', 'g_max', 'g_min', 'g_mean', 'g_std', 'num']].copy()

    #5 模型训练
    rf = RandomForestClassifier(random_state=42)
    rf.fit(x_train, y_train)

    #6 模型评估与验证
    def Fscore(y_true, y_predict):
        #
        # # 函数：测评标准
        #
        P = precision_score(y_true, y_predict, average='macro')
        R = recall_score(y_true, y_predict, average='macro')
        Fbeta = 100 * 5 * P * R / (2 * P + 3 * R)
        return Fbeta
    y_predict = rf.predict(x_validation)
    P = precision_score(y_validation, y_predict, average='macro')
    R = recall_score(y_validation, y_predict, average='macro')
    Fbeta = Fscore(y_validation, y_predict)
    C = confusion_matrix(y_validation, y_predict)
    report = classification_report(y_validation, y_predict)

    #7 模型预测
    #7.1 用全部训练数据训练模型
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    #7.2 重新预测
    y_out = rf.predict(x_test)
    test_data['pred_label'] = y_out
    #test_data[['id', 'v_max', 'v_min', 'v_mean', 'v_std', 'g_max', 'g_min', 'g_mean', 'g_std', 'num', 'pred_label']].head()
    #test_data.pred_label.value_counts()

    #8 结果提交
    ind = test_data[['id', 'pred_label']].pred_label == '0'
    out = test_data[['id', 'pred_label']][ind]
    ############################
    # 按提交要求将预测标签写入csv
    submission = pd.DataFrame({'id':out['id']})
    # submission.to_csv('src/step1/ground_truth/test_prediction.csv', index=False)
    submission.to_csv('/home/aistudio/work/test_prediction.csv', index=False)


# In[ ]:


getPrediction()


# 请点击[此处](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576)查看本环境基本用法.  <br>
# Please click [here ](https://ai.baidu.com/docs#/AIStudio_Project_Notebook/a38e5576) for more detailed instructions. 
