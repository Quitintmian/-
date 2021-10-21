#!/usr/bin/env python
# coding: utf-8

# **本次实验任务**
# 
# 以西瓜数据集3.0为例，使用朴素贝叶斯分类算法建立一个模型，并根据朴素贝叶斯分类算法流程对模型进行训练，得到一个能够准确对西瓜好坏进行识别的模型。
# 
# **提交要求**
# 
# （1）文本框插入“朴素贝叶斯分类算法实现代码”；（2）模型运行结果；（3）学习过程的心得体会、问题、建议（可参考亮考帮形式撰写）；（4）上传源代码。
# 
# 这三个类适用的分类场景各不相同，主要根据数据类型来进行模型的选择。一般来说，如果样本特征的分布大部分是连续值，使用GaussianNB会比较好。如果样本特征的分布大部分是多元离散值，使用MultinomialNB比较合适。而如果样本特征是二元离散值或者很稀疏的多元离散值，应该使用BernoulliNB。
# 
# **参考资料**
# 
# [1] https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
# 
# [2] https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# 
# [3] https://www.studyai.cn/modules/naive_bayes.html
# 
# [4] https://scikit-learn.org/stable/modules/naive_bayes.html
# 
# [5] https://zhuanlan.zhihu.com/p/117230627

# In[2]:


# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wednesday October 20 14:33:40 2021

@author: Xing-Rong Fan
"""


# In[34]:


import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from io import StringIO
# import subprocess


# **西瓜数据集3.0**
# 
# 编号,色泽,根蒂,敲声,纹理,脐部,触感,密度,含糖率,好瓜
# 
# 1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,0.697,0.46,是
# 
# 2,乌黑,蜷缩,沉闷,清晰,凹陷,硬滑,0.774,0.376,是
# 
# 3,乌黑,蜷缩,浊响,清晰,凹陷,硬滑,0.634,0.264,是
# 
# 4,青绿,蜷缩,沉闷,清晰,凹陷,硬滑,0.608,0.318,是
# 
# 5,浅白,蜷缩,浊响,清晰,凹陷,硬滑,0.556,0.215,是
# 
# 6,青绿,稍蜷,浊响,清晰,稍凹,软粘,0.403,0.237,是
# 
# 7,乌黑,稍蜷,浊响,稍糊,稍凹,软粘,0.481,0.149,是
# 
# 8,乌黑,稍蜷,浊响,清晰,稍凹,硬滑,0.437,0.211,是
# 
# 9,乌黑,稍蜷,沉闷,稍糊,稍凹,硬滑,0.666,0.091,否
# 
# 10,青绿,硬挺,清脆,清晰,平坦,软粘,0.243,0.267,否
# 
# 11,浅白,硬挺,清脆,模糊,平坦,硬滑,0.245,0.057,否
# 
# 12,浅白,蜷缩,浊响,模糊,平坦,软粘,0.343,0.099,否
# 
# 13,青绿,稍蜷,浊响,稍糊,凹陷,硬滑,0.639,0.161,否
# 
# 14,浅白,稍蜷,沉闷,稍糊,凹陷,硬滑,0.657,0.198,否
# 
# 15,乌黑,稍蜷,浊响,清晰,稍凹,软粘,0.36,0.37,否
# 
# 16,浅白,蜷缩,浊响,模糊,平坦,硬滑,0.593,0.042,否
# 
# 17,青绿,蜷缩,沉闷,稍糊,稍凹,硬滑,0.719,0.103,否
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/193838cc6379444c96b8a64dc07acb4b889a57f665ae4d3f89e6cb654737f670)
# 

# In[35]:



# 读入数据：人工处理特征
def createDataSet_handCrafted():
    ''' 数据读入 '''
    rawData = StringIO(
    """
    编号,色泽,根蒂,敲声,纹理,脐部,触感,密度,含糖率,好瓜
    1,1,1,0.5,1,1,1,0.697,0.46,1
    2,0,1,1,1,1,1,0.774,0.376,1
    3,0,1,0.5,1,1,1,0.634,0.264,1
    4,1,1,1,1,1,1,0.608,0.318,1
    5,0.5,1,0.5,1,1,1,0.556,0.215,1
    6,1,0.5,0.5,1,0.5,2,0.403,0.237,1
    7,0,0.5,0.5,0.5,0.5,2,0.481,0.149,1
    8,0,0.5,0.5,0.5,0.5,1,0.437,0.211,1
    9,0,0.5,1,1,0.5,1,0.666,0.091,0
    10,1,0,0,0.5,0,2,0.243,0.267,0
    11,0.5,0,0,0,0,1,0.245,0.057,0
    12,0.5,1,0.5,0,0,2,0.343,0.099,0
    13,1,0.5,0.5,0.5,1,1,0.639,0.161,0
    14,0.5,0.5,1,0.5,1,1,0.657,0.198,0
    15,0,0.5,0.5,1,0.5,2,0.36,0.37,0
    16,0.5,1,0.5,0,0,1,0.593,0.042,0
    17,1,1,1,0.5,0.5,1,0.719,0.103,0
""")
    df = pd.read_csv(rawData, sep=",") 
    return df


# In[36]:


# 读入数据：自动批量处理特征
def createDataSet_automated():
    ''' 数据读入 '''
    rawData = StringIO(
    """
编号,色泽,根蒂,敲声,纹理,脐部,触感,密度,含糖率,好瓜
1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,0.697,0.46,是
2,乌黑,蜷缩,沉闷,清晰,凹陷,硬滑,0.774,0.376,是
3,乌黑,蜷缩,浊响,清晰,凹陷,硬滑,0.634,0.264,是
4,青绿,蜷缩,沉闷,清晰,凹陷,硬滑,0.608,0.318,是
5,浅白,蜷缩,浊响,清晰,凹陷,硬滑,0.556,0.215,是
6,青绿,稍蜷,浊响,清晰,稍凹,软粘,0.403,0.237,是
7,乌黑,稍蜷,浊响,稍糊,稍凹,软粘,0.481,0.149,是
8,乌黑,稍蜷,浊响,清晰,稍凹,硬滑,0.437,0.211,是
9,乌黑,稍蜷,沉闷,稍糊,稍凹,硬滑,0.666,0.091,否
10,青绿,硬挺,清脆,清晰,平坦,软粘,0.243,0.267,否
11,浅白,硬挺,清脆,模糊,平坦,硬滑,0.245,0.057,否
12,浅白,蜷缩,浊响,模糊,平坦,软粘,0.343,0.099,否
13,青绿,稍蜷,浊响,稍糊,凹陷,硬滑,0.639,0.161,否
14,浅白,稍蜷,沉闷,稍糊,凹陷,硬滑,0.657,0.198,否
15,乌黑,稍蜷,浊响,清晰,稍凹,软粘,0.36,0.37,否
16,浅白,蜷缩,浊响,模糊,平坦,硬滑,0.593,0.042,否
17,青绿,蜷缩,沉闷,稍糊,稍凹,硬滑,0.719,0.103,否
""")
    df = pd.read_csv(rawData, sep=",") 
    return df


# In[37]:


# 可测试不同编码特征工程下结果的差异
df_handCrafted = createDataSet_handCrafted() # 手工将西瓜数据集特征变量重编码处理
df_automated = createDataSet_automated()     # 基于LabelEncoder的类别变量重编码
df = df_automated
df.head()


# In[38]:


# 基于LabelEncoder的类别变量重编码
class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode
    def fit(self,X,y=None):
        return self # not relevant here
    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output
    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[39]:


df = MultiColumnLabelEncoder(columns=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '好瓜']).fit_transform(df)
print(df)


# In[40]:


feature_names = ['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']
target_names = ['是', '否']
feature = df[feature_names]
label = df['好瓜']
X = feature
y = label
print(X)
print(y)


# In[41]:


def evaluationResults(modelName, X, y, n_splits_cv):
    '''
    函数功能：采用k-fold cv计算给定模型的accuracy、precision、recall和f1
    函数输入：（1）模型名称：modelName = {'BNB':'BernoulliNB', 'MNB':'MultinomialNB', 'GNB':'GaussianNB'}
             （2）数据：X, y
             （3）划分折数：n_splits_cv
    函数输出：返回模型accuracy、precision、recall和f1
    '''
    evalResults = {}
    if modelName == 'BernoulliNB':
        model = BernoulliNB()
    elif modelName == 'MultinomialNB':
        model = MultinomialNB()
    elif modelName == 'GaussianNB':
        model = GaussianNB()
    else:
        print("您输入的模型有误！")
    bits = 3
    scores = cross_validate(model, X, y, cv=n_splits_cv, scoring=('accuracy', 'precision','recall','f1'), n_jobs=-1, return_train_score=True)
    evalResults[modelName+"\'s mean"] = [round(np.mean(scores['test_accuracy']),bits),                                                          round(np.mean(scores['test_precision']),bits),                                                          round(np.mean(scores['test_recall']),bits),                                                          round(np.mean(scores['test_f1']),bits)]
    evalResults[modelName+"\'s std"] =  [round(np.std(scores['test_accuracy']),bits),                                                          round(np.std(scores['test_precision']),bits),                                                         round(np.std(scores['test_recall']),bits),                                                          round(np.std(scores['test_f1']),bits)]
    model_eval_results = pd.DataFrame(evalResults,index=('ACC','P','R','F1'))                                   
    return model_eval_results


# In[42]:


n_splits_cv = 8
modelName = {'BNB':'BernoulliNB', 'MNB':'MultinomialNB', 'GNB':'GaussianNB'}
df_BNB = evaluationResults(modelName['BNB'], X, y, n_splits_cv)
print(df_BNB)
df_MNB = evaluationResults(modelName['MNB'], X, y, n_splits_cv)
print(df_MNB)
df_GNB = evaluationResults(modelName['GNB'], X, y, n_splits_cv)
print(df_GNB)
resultsTable = pd.concat([df_BNB, df_MNB, df_GNB],axis=1)
# print(resultsTable)

