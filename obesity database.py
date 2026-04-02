import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv('/Users/bibi/Desktop/Obesity_Data_clean.csv')

#零食频率
snacking_mapping = {
    'no': 0,
    'Sometimes': 1,   
    'Frequently': 2,
    'Always': 3
}
df['snacking_encoded'] = df['snacking'].map(snacking_mapping)

#饮酒频率   
alcohol_mapping = {
    'no': 0,
    'Sometimes': 1,
    'Frequently': 2,
    'Always': 3
}
df['alcohol_encoded'] = df['alcohol'].map(alcohol_mapping)

#是否有家族肥胖史
family_history_mapping = {
    'no': 0,
    'yes': 1,

}
df['family_history_encoded'] = df['family_history'].map(family_history_mapping)

#是否摄入高热量食物
high_cal_food_mapping = {
    'no': 0,
    'yes': 1,

}
df['high_cal_food_encoded'] = df['high_cal_food'].map(high_cal_food_mapping)


#公共交通

transport_mapping = {
    'Public_Transportation': 0,
    'Walking': 1,
    'Automobile': 2,
     'Bike': 3  

}
df['transport_encoded'] = df['transport'].map(transport_mapping)


#是否吸烟
smoking_mapping = {
    'no': 0,
    'yes': 1,

}
df['smoking_encoded'] = df['smoking'].map(smoking_mapping)

#删除列数
df = df.drop(['meals_per_day', 'veg_consumption','water_intake','calorie_monitor'], axis=1) 


#将数值取整数 四舍五入
df['age'] = df['age'].round(0).astype(int)
df['physical_activity'] = df['physical_activity'].round(0).astype(int)
df['screen_time'] = df['screen_time'].round(0).astype(int)

#将身高体重后面保留两位小数
df['height'] = df['height'].round(2)
df['weight'] = df['weight'].round(2)


print(df.head())

df.to_csv('/Users/bibi/Desktop/Obesity_Data_clean(1).csv', index=False, encoding='utf-8-sig')