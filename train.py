import numpy as np

true_stand_data=np.load('./true_array.npy')
false_stand_data=np.load('./false_array.npy')

sample_num={'true_num':true_stand_data.shape[0],'false_num':false_stand_data.shape[0]}
# 构造训练数据：
combin_train_data = np.concatenate((true_stand_data,false_stand_data),axis=0)
combin_target_data= np.concatenate((np.ones((sample_num['true_num'],1)),np.zeros((sample_num['false_num'],1))),axis=0)

print(combin_train_data.shape,combin_target_data.shape)



import tensorflow as tf

# 构建dataset，其实是把pandas数据转换成numpy数组进行转换的
dataset = tf.data.Dataset.from_tensor_slices((combin_train_data, combin_target_data))
class_weight = {
    0: (1 / sample_num['false_num'] * (sample_num['true_num'] + sample_num['false_num'])) / 2,
    1: (1 / sample_num['true_num'] * (sample_num['true_num'] + sample_num['false_num'])) / 2
}

# for features, label in dataset.take(1):
#     print('Features: {}, Label: {}'.format(features, label))

# Shuffle and batch the dataset.
print(dataset.output_shapes)
train_dataset = dataset.shuffle(combin_train_data.shape[0]).batch(1)


model = tf.keras.Sequential([
    tf.keras.layers.GRU(units=48,
          return_sequences=True,
          input_shape=(51, 7)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GRU(units=24),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1,input_shape=(24,), activation='sigmoid')
])
print(model.summary())

model.compile(optimizer='adam',
#             loss='tf.keras.losses.BinaryCrossentropy(from_logits=True)',
            loss='binary_crossentropy',
            metrics=['accuracy'])

model.fit(train_dataset, epochs=15,class_weight=class_weight)
model.save('GRU_predict.h5')






