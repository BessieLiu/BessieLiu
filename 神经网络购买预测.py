#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import matplotlib
from matplotlib.ticker import FuncFormatter
plt.rcParams['font.sans-serif']=['Arial Unicode MS']
 
import warnings
warnings.filterwarnings('ignore')
 
#读取订单数据和用户信息数据
import os


# In[9]:


os.chdir('C:/Users/李乔乔/Desktop/数据分析大赛/数据/')


# In[28]:


user_cp=pd.read_csv('user_province.csv',delimiter='t',engine='python')


# In[32]:



user_ct=pd.read_csv('user_cityp.csv',delimiter='t',engine='python')


# In[30]:


user_cp.info()


# In[33]:


user_cityp =pd.DataFrame(user_ct )
       


# In[38]:


plt.hist(user_cityp[ 'age_month' ])
        plt.show()


# In[42]:


import pandas as pd
       
import numpy as  np
       
import os
       
import matplotlib  as  mpl
       
import matplotlib.pyplot as  plt


# In[44]:


plt.hist(user_cityp[ 'age_month' ])
 


# In[45]:


plt.boxplot(user_cp[  'age_month' ])


# In[67]:


from collections import Counter

a1 = ['主营业务：', 'O2O', '车主服务', '驾考培训', '互联网驾考', '汽车交通支撑服务', '汽车交通支撑服务', '汽车交通支撑服务', '汽车交通支撑服务', '驾考培训', '驾考培训', '主营业务：']
# 统计词频
result = Counter(a)
print(result)
# 排序
d = sorted(result.items(), key=lambda x: x[1], reverse=True)
print(d)


# In[71]:


a=user_ct.iloc[:,4]
from collections import Counter

result = Counter(a)
print(result)
# 排序
d = sorted(result.items(), key=lambda x: x[1], reverse=True)
print(d)


# In[72]:


counter1={'重庆': 12390, '成都': 3596, '运城': 3538, '广州': 3165, '北京': 2576, '洛阳': 2439, '保定': 2160, '泉州': 1917, '深圳': 1880, '上海': 1650, '郑州': 1639, '西安': 1637, '邯郸': 1604, '东莞': 1484, '衡阳': 1293, '福州': 1282, '贵阳': 1153, '杭州': 1119, '三门峡': 1105, '佛山': 1091, '石家庄': 1007, '天津': 924, '南昌': 885, '临汾': 873, '邢台': 862, '长沙': 852, '苏州': 828, '咸阳': 827, '济南': 793, '济宁': 783, '青岛': 781, '徐州': 732, '合肥': 732, '晋城': 726, '武汉': 721, '唐山': 675, '临沂': 647, '中山': 642, '南京': 637, '德州': 616, '潍坊': 585, '沧州': 572, '惠州': 569, '沈阳': 567, '庆阳': 532, '太原': 519, '厦门': 505, '温州': 460, '烟台': 435, '驻马店': 433, '衡水': 431, '哈尔滨': 416, '菏泽': 413, '乌鲁木齐': 412, '资阳': 410, '廊坊': 399, '长春': 395, '无锡': 392, '长治': 365, '泰安': 359, '濮阳': 355, '大连': 348, '宁波': 343, '南宁': 340, '阜阳': 315, '南阳': 312, '呼和浩特': 307, '开封': 307, '聊城': 304, '银川': 301, '榆林': 300, '金华': 299, '常州': 286, '许昌': 285, '南通': 285, '汕头': 281, '昆明': 281, '达州': 279, '赣州': 278, '南充': 277, '渭南': 272, '张家口': 271, '新乡': 269, '淄博': 266, '湛江': 266, '台州': 259, '吕梁': 258, '赤峰': 255, '商丘': 253, '安阳': 252, '周口': 252, '盐城': 249, '连云港': 248, '梅州': 247, '包头': 243, '枣庄': 241, '滨州': 238, '江门': 238, '晋中': 236, '遵义': 233, '肇庆': 232, '漳州': 230, '日照': 227, '岳阳': 225, '鄂尔多斯': 224, '株洲': 222, '九江': 218, '绍兴': 217, '兰州': 215, '东营': 214, '铁岭': 214, '亳州': 213, '桂林': 211, '襄阳': 210, '扬州': 20}


# In[92]:



key1=counter1.keys()
value1=counter1.values()
print(key1)
print(value1)


# In[93]:


df = pd.DataFrame({'city': ['重庆', '成都', '运城', '广州', '北京', '洛阳', '保定', '泉州', '深圳', '上海', '郑州', '西安', '邯郸', '东莞', '衡阳', '福州', '贵阳', '杭州', '三门峡', '佛山', '石家庄', '天津', '南昌', '临汾', '邢台', '长沙', '苏州', '咸阳', '济南', '济宁', '青岛', '徐州', '合肥', '晋城', '武汉', '唐山', '临沂', '中山', '南京', '德州', '潍坊', '沧州', '惠州', '沈阳', '庆阳', '太原', '厦门', '温州', '烟台', '驻马店', '衡水', '哈尔滨', '菏泽', '乌鲁木齐', '资阳', '廊坊', '长春', '无锡', '长治', '泰安', '濮阳', '大连', '宁波', '南宁', '阜阳', '南阳', '呼和浩特', '开封', '聊城', '银川', '榆林', '金华', '常州', '许昌', '南通', '汕头', '昆明', '达州', '赣州', '南充', '渭南', '张家口', '新乡', '淄博', '湛江', '台州', '吕梁', '赤峰', '商丘', '安阳', '周口', '盐城', '连云港', '梅州', '包头', '枣庄', '滨州', '江门', '晋中', '遵义', '肇庆', '漳州', '日照', '岳阳', '鄂尔多斯', '株洲', '九江', '绍兴', '兰州', '东营', '铁岭', '亳州', '桂林', '襄阳', '扬州'], 'counter': [12390, 3596, 3538, 3165, 2576, 2439, 2160, 1917, 1880, 1650, 1639, 1637, 1604, 1484, 1293, 1282, 1153, 1119, 1105, 1091, 1007, 924, 885, 873, 862, 852, 828, 827, 793, 783, 781, 732, 732, 726, 721, 675, 647, 642, 637, 616, 585, 572, 569, 567, 532, 519, 505, 460, 435, 433, 431, 416, 413, 412, 410, 399, 395, 392, 365, 359, 355, 348, 343, 340, 315, 312, 307, 307, 304, 301, 300, 299, 286, 285, 285, 281, 281, 279, 278, 277, 272, 271, 269, 266, 266, 259, 258, 255, 253, 252, 252, 249, 248, 247, 243, 241, 238, 238, 236, 233, 232, 230, 227, 225, 224, 222, 218, 217, 215, 214, 214, 213, 211, 210, 20]})


# In[98]:


df[:21]


# In[101]:


import xlrd
 
file_name = "user_visit_login_result.xls"

 


# In[102]:


sum_need = pd.read_excel('sum_need.xlsx')


# In[104]:


data1 = sum_need.head(7)
data1


# In[105]:


def check_missing_data(df):
    flag=df.isna().sum().any()
    if flag==True:
        total = df.isnull().sum()
        percent = (df.isnull().sum())/(df.isnull().count()*100)
        output = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        data_type = []
        # written by MJ Bahmani
        for col in df.columns:
            dtype = str(df[col].dtype)
            data_type.append(dtype)
        output['Types'] = data_type
        return(np.transpose(output))
    else:
        return(False)


# In[106]:


print(check_missing_data(sum_need))


# In[107]:



get_ipython().run_line_magic('matplotlib', 'inline')


# In[116]:


data = pd.read_csv("C:\\Users\\李乔乔\\Desktop\\数据分析大赛\\数据\\bank\\bank-full.csv",sep = ';')
test = pd.read_csv("C:\\Users\\李乔乔\\Desktop\\数据分析大赛\\数据\\bank\\bank.csv",sep = ';')
data.head(7)


# In[117]:


test.head(7)


# In[123]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

def get_y_train():
    y_train = np.array(data['y'])
    y_train = np.where(y_train == 'yes', 0, 1) #将预测值转换成01
    return y_train
 
def get_y_test():
    y_test = np.array(test['y'])
    y_test = np.where(y_test == 'yes', 0, 1) #将预测值转换成01
    return y_test
 
def get_X_train():
    oh_data = pd.get_dummies(data) #对非数值数据进行ont-hot编码
    columns_size = oh_data.columns.size
    X_train = oh_data.iloc[:,0:columns_size-2] #取特征
    X_train = preprocessing.scale(X_train) #归一化
    return X_train
 
def get_X_test():
    oh_test = pd.get_dummies(test) #对非数值数据进行ont-hot编码
    columns_size = oh_test.columns.size
    X_test = oh_test.iloc[:,0:columns_size-2] #取特征
    X_test= preprocessing.scale(X_test) #归一化
    return X_test
X_test = get_X_test()
X_train = get_X_train()
y_test = get_y_test()
y_train = get_y_train()


# In[152]:



from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential
from keras import optimizers
from sklearn import metrics
 
 
model = Sequential()
model.add(Dense(51, input_dim = 51, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer = optimizers.Adam(lr = 0.006),loss = 'binary_crossentropy',metrics = ['accuracy'])

history = model.fit(X_train, y_train,epochs = 20,  batch_size = 512, validation_data = (X_test, y_test))


# In[147]:



y_pred = model.predict_classes(X_test, batch_size = 20, verbose = 1)
print(y_pred)




# In[148]:


target_names = ['1', '0']
print(metrics.classification_report(y_test, y_pred,
    target_names = target_names))



# In[150]:


df1 = pd.DataFrame(y_pred)
df1


# In[153]:




history_dict = history.history
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test accuracy: ', accuracy)


# In[159]:


history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc = history_dict['accuracy']
epochs = range(1, len(acc) + 1)


# In[158]:


print(history_dict)


# In[160]:


plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']


# In[161]:


plt.plot(epochs, acc_values, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()


# In[168]:


df3= pd.DataFrame(y_pred,y_test)
df3
df3.to_csv("out_path.csv")


# In[165]:


X_test


# In[169]:


df2= pd.DataFrame(y_pred,y_test)
df2


# In[173]:


df3= pd.DataFrame(y_pred,y_test)
df3
df3.to_csv("out_path2.csv",header=False)


# In[174]:


print(y_test)


# In[ ]:




