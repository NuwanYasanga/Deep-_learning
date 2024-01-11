import pandas as pd
import numpy as np
import tensorflow as tf
import os
import random
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense



data_dir_desktop = 'C:/Research Activities/Datasets/BB-MAS_Dataset/Desktop_data/Desktop_all_users_FT/'

user_desktop = os.listdir(data_dir_desktop)

all_other_users = []

for i in (range(len(user_desktop))):
    user_file_desktop = user_desktop[i]
    curr_user_ind_desktop = int(user_file_desktop[user_file_desktop.find('_')+1:user_file_desktop.find('.')])
    if curr_user_ind_desktop ==1:
        data_frame_desktop_bf1 = pd.read_csv(data_dir_desktop +user_file_desktop, header=0)
        data_frame_desktop_bf1 = data_frame_desktop_bf1[['Keys', 'F1','F2','F3','F4','Distance','Hands']]
        data_frame_desktop_bf2 = data_frame_desktop_bf1[(data_frame_desktop_bf1['F1']>(-5000)) & (data_frame_desktop_bf1['F2']>(-5000))]
        data_frame_desktop_bf3 = data_frame_desktop_bf2[(data_frame_desktop_bf2['F1']<(5000)) & (data_frame_desktop_bf2['F2']<(5000)) & (data_frame_desktop_bf2['F3']<(5000)) & (data_frame_desktop_bf2['F4']<(5000))]
        desktop_user_final = data_frame_desktop_bf3[(data_frame_desktop_bf3['Hands']!='LR') & (data_frame_desktop_bf3['Distance'].isin([0,1,2,3]))]
        desktop_user_final.insert(loc = len(desktop_user_final.columns),column = 'User_type',value = 1)
        user_train_set = desktop_user_final.sample(frac=0.8, random_state=42)
        user_test_set = desktop_user_final.drop(user_train_set.index).reset_index(drop=True)

    else :
        data_frame_desktop_bf1 = pd.read_csv(data_dir_desktop +user_file_desktop, header=0)
        data_frame_desktop_bf1 = data_frame_desktop_bf1[['Keys', 'F1','F2','F3','F4','Distance','Hands']]
        data_frame_desktop_bf2 = data_frame_desktop_bf1[(data_frame_desktop_bf1['F1']>(-5000)) & (data_frame_desktop_bf1['F2']>(-5000))]
        data_frame_desktop_bf3 = data_frame_desktop_bf2[(data_frame_desktop_bf2['F1']<(5000)) & (data_frame_desktop_bf2['F2']<(5000)) & (data_frame_desktop_bf2['F3']<(5000)) & (data_frame_desktop_bf2['F4']<(5000))]
        desktop_final = data_frame_desktop_bf3[(data_frame_desktop_bf3['Hands']!='LR') & (data_frame_desktop_bf3['Distance'].isin([0,1,2,3]))]
        desktop_final.insert(loc = len(desktop_final.columns),column = 'User_type',value = 0)
        all_other_users.append(desktop_final)

all_other_users_df = pd.concat(all_other_users, ignore_index=True)
all_train_set = all_other_users_df.sample(frac=0.8, random_state=42)
all_test_set = all_other_users_df.drop(all_train_set.index).reset_index(drop=True)

final_train_set = user_train_set.append(all_train_set)
final_test_set = user_test_set.append(all_test_set)

x_train = final_train_set[{'F1','F2','F3','F4','Distance'}]
x_test = final_test_set[{'F1','F2','F3','F4','Distance'}]

y_train = final_train_set['User_type']
y_test = final_test_set['User_type']

resample = SMOTE(random_state=42)
x_train_res, y_train_res = resample.fit_resample(x_train, y_train)

num_features = 5
num_timesteps = x_train_res.shape[1] // num_features 


x_train_3d = x_train_res.values.reshape((-1, num_timesteps, num_features))
x_test_3d = x_test.values.reshape((-1, num_timesteps, num_features))
y_train = y_train_res.values
y_test = y_test.values

model = Sequential()
model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=(x_train_3d.shape[1], x_train_3d.shape[2])))
model.add(Bidirectional(LSTM(units=20)))
model.add(Dense(1, activation='sigmoid'))  # Adjust the number of units and activation function

 #Compile the model
model.compile(loss='categorical_crossentropy',  # or another appropriate loss function
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
x_train_tensor = tf.convert_to_tensor(x_train_3d, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train_res, dtype=tf.float32)

history = model.fit(x_train_tensor, y_train_tensor, epochs=3, batch_size=64, validation_split=0.2, verbose=1)

predictions = model.predict(x_test_3d)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = y_test

from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(true_classes, predicted_classes)
report = classification_report(true_classes, predicted_classes)

print(f'Accuracy: {accuracy}')
print(report)

#print(len(desktop_user_final))
#print(len(user_train_set))
#print(len(user_test_set))

#print(len(all_other_users_df))
#print(len(all_train_set))
#print(len(all_test_set))

#print(len(final_train_set))
#print(len(final_test_set))

#print(x_train.head(10))
#print("---------------")
#print(y_train.head(10))
    
