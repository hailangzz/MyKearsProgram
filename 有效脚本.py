### model V8
import keras
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import copy

from keras.layers.recurrent import GRU
from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Concatenate,Dropout
import copy
import numpy as np
import tensorflow as tf

true_stand_data=np.load(r'.//data3//true_array.npy')
false_stand_data=np.load(r'.//data3//false_array.npy')


true_stand_data=true_stand_data[:,:,:,0]
false_stand_data=false_stand_data[:,:,:,0]

true_stand_data=true_stand_data.swapaxes(1, 2)
false_stand_data=false_stand_data.swapaxes(1, 2)

sample_num={'true_num':true_stand_data.shape[0],'false_num':false_stand_data.shape[0]}
class_weight = {
                0: (1 / sample_num['false_num'] * (sample_num['true_num'] + sample_num['false_num'])) / 2,
                1: (1 / sample_num['true_num'] * (sample_num['true_num'] + sample_num['false_num'])) / 2
                }

true_sample_channel_1 = []
false_sample_channel_1 = []


for sample_number in range(true_stand_data.shape[0]):
    true_sample_channel_1.append(copy.deepcopy(true_stand_data[sample_number,:,:]))
    
    
for sample_number in range(false_stand_data.shape[0]):
    false_sample_channel_1.append(copy.deepcopy(false_stand_data[sample_number,:,:]))
    

combin_train_data1 = np.concatenate((true_sample_channel_1,false_sample_channel_1),axis=0)

combin_target_data= np.concatenate((np.ones((sample_num['true_num'],1)),np.zeros((sample_num['false_num'],1))),axis=0)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
combin_target_data=ohe.fit_transform(combin_target_data).toarray()

# Model 1
in_1 = Input(shape=(7,110))
x_480_1 = GRU(units=480,return_sequences=True,input_shape=(7,110))(in_1)
x_100_1 = GRU(units=100)(x_480_1)
x_10_1 =Dense(10, activation='tanh')(x_100_1)

model_final_dense_out = Dense(2, activation='softmax')(x_10_1)

model = Model(inputs=in_1, outputs=model_final_dense_out)
print(model.summary())


import os
if 'GRU_predict_v8.h5' in os.listdir('./'):
    model = load_model("./GRU_predict_v8.h5")

callbacks_list = [    
    keras.callbacks.EarlyStopping(
        monitor='accuracy', 
        patience=30
    ),
    # 保存模型
    keras.callbacks.ModelCheckpoint(
    filepath = 'GRU_predict_v8.h5', 
    monitor='loss', 
    save_best_only=True),
    
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.8,
        patience=5, 
        mode='auto',
        min_lr=0.00003)
]


model.compile(loss='categorical_crossentropy', #continu together
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy']
             
             )

model.fit(x=combin_train_data1,
          y=combin_target_data,
          batch_size=360,
          epochs=100,
          shuffle=True,
          verbose=1,
          class_weight=class_weight,
          callbacks=callbacks_list      
         )


### model V10_2_3

import keras
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import copy

from keras.layers.recurrent import GRU
from keras.models import Model
from keras.layers import Input, Convolution2D,Convolution1D,MaxPooling1D, Flatten, Dense, Concatenate,Dropout
import copy
import numpy as np
import tensorflow as tf


true_stand_data=np.load(r'.//data3//true_array.npy')
false_stand_data=np.load(r'.//data3//false_array.npy')

true_stand_data=true_stand_data[:,:,[1,2,6],0]
false_stand_data=false_stand_data[:,:,[1,2,6],0]


sample_num={'true_num':true_stand_data.shape[0],'false_num':false_stand_data.shape[0]}
class_weight = {
                0: (1 / sample_num['false_num'] * (sample_num['true_num'] + sample_num['false_num'])) / 2,
                1: (1 / sample_num['true_num'] * (sample_num['true_num'] + sample_num['false_num'])) / 2
                }

true_sample_channel_1 = []
false_sample_channel_1 = []


for sample_number in range(true_stand_data.shape[0]):
    true_sample_channel_1.append(copy.deepcopy(true_stand_data[sample_number,:,:]))
    
for sample_number in range(false_stand_data.shape[0]):
    false_sample_channel_1.append(copy.deepcopy(false_stand_data[sample_number,:,:]))
    
    
combin_train_data1 = np.concatenate((true_sample_channel_1,false_sample_channel_1),axis=0)

combin_target_data= np.concatenate((np.ones((sample_num['true_num'],1)),np.zeros((sample_num['false_num'],1))),axis=0)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
combin_target_data=ohe.fit_transform(combin_target_data).toarray()    
    
# Model 1
in_1 = Input(shape=(110,3))
C_1 = Convolution1D(filters=12,kernel_size=3,input_shape=(110,3),padding='same')(in_1)
C_1 = Convolution1D(filters=12,kernel_size=3,padding='same')(C_1)
C_1 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_1)

C_1 = Convolution1D(filters=36,kernel_size=3,padding='same')(C_1)
C_1 = Convolution1D(filters=36,kernel_size=3,padding='same')(C_1)
C_1 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_1)

C_1 = Convolution1D(filters=108,kernel_size=3,padding='same')(C_1)
C_1 = Convolution1D(filters=108,kernel_size=3,padding='same')(C_1)
C_1 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_1)

C_1 = Convolution1D(filters=240,kernel_size=3,padding='same')(C_1)
C_1 = Convolution1D(filters=240,kernel_size=3,padding='same')(C_1)
C_1 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_1)


x_middle_concat = GRU(units=480,return_sequences=True)(C_1)
x_middle_concat = GRU(units=100)(x_middle_concat)

model_final_dense_out = Dense(2, activation='softmax')(x_middle_concat)

model = Model(inputs=in_1, outputs=model_final_dense_out)
print(model.summary())    
    
import os
if 'GRU_predict_v10_3.h5' in os.listdir('./'):
    model = load_model("./GRU_predict_v10_3.h5")

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='accuracy', 
        patience=30
    ),
    # 保存模型
    keras.callbacks.ModelCheckpoint(
    filepath = 'GRU_predict_v10_3.h5', 
    monitor='loss', 
    save_best_only=True),
    
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.8,
        patience=5, 
        mode='auto',
        min_lr=0.00003)
]


model.compile(loss='categorical_crossentropy', #continu together
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy']
             
             )

model.fit(x=combin_train_data1,
          y=combin_target_data,
          batch_size=360,
          epochs=100,
          shuffle=True,
          verbose=1,
          class_weight=class_weight,
          callbacks=callbacks_list 
         )

    
    
    
    
    
### model V10_2_1

import keras
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import GRU
from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import copy

from keras.layers.recurrent import GRU
from keras.models import Model
from keras.layers import Input, Convolution2D,Convolution1D,MaxPooling1D, Flatten, Dense, Concatenate,Dropout
import copy
import numpy as np
import tensorflow as tf


true_stand_data=np.load(r'.//data3//true_array.npy')
false_stand_data=np.load(r'.//data3//false_array.npy')

true_stand_data=true_stand_data[:,:,:,0]
false_stand_data=false_stand_data[:,:,:,0]

sample_num={'true_num':true_stand_data.shape[0],'false_num':false_stand_data.shape[0]}
class_weight = {
                0: (1 / sample_num['false_num'] * (sample_num['true_num'] + sample_num['false_num'])) / 2,
                1: (1 / sample_num['true_num'] * (sample_num['true_num'] + sample_num['false_num'])) / 2
                }

true_sample_channel_1 = []
true_sample_channel_2 = []
true_sample_channel_3 = []

false_sample_channel_1 = []
false_sample_channel_2 = []
false_sample_channel_3 = []


for sample_number in range(true_stand_data.shape[0]):
    true_sample_channel_1.append(copy.deepcopy(true_stand_data[sample_number,:,1]))
    true_sample_channel_2.append(copy.deepcopy(true_stand_data[sample_number,:,2]))
    true_sample_channel_3.append(copy.deepcopy(true_stand_data[sample_number,:,6]))
    
for sample_number in range(false_stand_data.shape[0]):
    false_sample_channel_1.append(copy.deepcopy(false_stand_data[sample_number,:,1]))
    false_sample_channel_2.append(copy.deepcopy(false_stand_data[sample_number,:,2]))
    false_sample_channel_3.append(copy.deepcopy(false_stand_data[sample_number,:,6]))
    
combin_train_data1 = np.concatenate((true_sample_channel_1,false_sample_channel_1),axis=0)
combin_train_data2 = np.concatenate((true_sample_channel_2,false_sample_channel_2),axis=0)
combin_train_data3 = np.concatenate((true_sample_channel_3,false_sample_channel_3),axis=0)

combin_target_data= np.concatenate((np.ones((sample_num['true_num'],1)),np.zeros((sample_num['false_num'],1))),axis=0)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
combin_target_data=ohe.fit_transform(combin_target_data).toarray()    
    
    
# Model 1
in_1 = Input(shape=(110,1))
C_1 = Convolution1D(filters=12,kernel_size=3,input_shape=(110,1),padding='same')(in_1)
C_1 = Convolution1D(filters=12,kernel_size=3,padding='same')(C_1)
C_1 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_1)

C_1 = Convolution1D(filters=24,kernel_size=3,padding='same')(C_1)
C_1 = Convolution1D(filters=24,kernel_size=3,padding='same')(C_1)
C_1 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_1)

C_1 = Convolution1D(filters=48,kernel_size=3,padding='same')(C_1)
C_1 = Convolution1D(filters=48,kernel_size=3,padding='same')(C_1)
C_1 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_1)

C_1 = Convolution1D(filters=96,kernel_size=3,padding='same')(C_1)
C_1 = Convolution1D(filters=96,kernel_size=3,padding='same')(C_1)
# C_1 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_1)
# C_1 = Flatten()(C_1)

# # Model 1
in_2 = Input(shape=(110,1))
C_2 = Convolution1D(filters=12,kernel_size=3,input_shape=(110,1),padding='same')(in_2)
C_2 = Convolution1D(filters=12,kernel_size=3,padding='same')(C_2)
C_2 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_2)

C_2 = Convolution1D(filters=24,kernel_size=3,padding='same')(C_2)
C_2 = Convolution1D(filters=24,kernel_size=3,padding='same')(C_2)
C_2 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_2)

C_2 = Convolution1D(filters=48,kernel_size=3,padding='same')(C_2)
C_2 = Convolution1D(filters=48,kernel_size=3,padding='same')(C_2)
C_2 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_2)

C_2 = Convolution1D(filters=96,kernel_size=3,padding='same')(C_2)
C_2 = Convolution1D(filters=96,kernel_size=3,padding='same')(C_2)
# C_2 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_2)
# C_2 = Flatten()(C_2)

in_3 = Input(shape=(110,1))
C_3 = Convolution1D(filters=12,kernel_size=3,input_shape=(110,1),padding='same')(in_3)
C_3 = Convolution1D(filters=12,kernel_size=3,padding='same')(C_3)
C_3 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_3)

C_3 = Convolution1D(filters=24,kernel_size=3,padding='same')(C_3)
C_3 = Convolution1D(filters=24,kernel_size=3,padding='same')(C_3)
C_3 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_3)

C_3 = Convolution1D(filters=48,kernel_size=3,padding='same')(C_3)
C_3 = Convolution1D(filters=48,kernel_size=3,padding='same')(C_3)
C_3 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_3)

C_3 = Convolution1D(filters=96,kernel_size=3,padding='same')(C_3)
C_3 = Convolution1D(filters=96,kernel_size=3,padding='same')(C_3)
# C_3 = MaxPooling1D(pool_size=2,strides=2, padding='valid')(C_3)
# C_3 = Flatten()(C_3)

x_middle_concat = Concatenate(axis=-1)([C_1,C_2,C_3])

x_middle_concat = GRU(units=288,return_sequences=True)(x_middle_concat)
x_middle_concat = GRU(units=100)(x_middle_concat)
# x_middle_concat = Dense(100, activation='tanh')(x_middle_concat)
# x_middle_concat = Flatten()(x_middle_concat)
model_final_dense_out = Dense(2, activation='softmax')(x_middle_concat)


model = Model(inputs=[in_1,in_2,in_3], outputs=model_final_dense_out)
print(model.summary())

    
    
import os
if 'GRU_predict_v10_2_1.h5' in os.listdir('./'):
    model = load_model("./GRU_predict_v10_2_1.h5")

callbacks_list = [
    keras.callbacks.EarlyStopping(
        monitor='accuracy', 
        patience=30
    ),
    # 保存模型
    keras.callbacks.ModelCheckpoint(
    filepath = 'GRU_predict_v10_2_1.h5', 
    monitor='loss', 
    save_best_only=True),
    
    keras.callbacks.ReduceLROnPlateau(
        monitor='loss', 
        factor=0.8,
        patience=5, 
        mode='auto',
        min_lr=0.00003)
]


model.compile(loss='categorical_crossentropy', #continu together
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy']             
             )

model.fit(x=[combin_train_data1,combin_train_data2,combin_train_data3],
          y=combin_target_data,
          batch_size=360,
          epochs=100,
          shuffle=True,
          verbose=1,
          class_weight=class_weight,
          callbacks=callbacks_list 
         )
         


