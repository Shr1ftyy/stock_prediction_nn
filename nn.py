'''
 ----- To Do: -----
 -  Figure out how to split data #########################################
    into Validation and training, testing (if not
    already done) - EDIT - >>>REVISIT SENTDEX VIDEO<<<

 -  After Validation and loss is all tuned and fixed,
    move on with inputting more features into the 
    Neural Network. Figure out how it should be shaped
    to input them into the network.
 
 -  Look more into Deep Reinforcement Learning 
 - News Sentiment Analysis (Natural Language Processing, make seperate neural net.
   or using processing library like BERT) 

 - Normalization vs. Scaling

 - 

'''

import math
import os
import random
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

import data_api as data

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75, allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

VERSION = 4
NAME = f'stock_model_v{VERSION}_{data.selection}_{time.time()}.h5'

#state variables    
window = 60
row = 1
length = len(data.stat)-1
PREDICT_SIZE = 100
VALIDATION_PCT = 0.15

EPOCHS = 100
BATCH_SIZE = 32



stock_open=data.stock_open
high=data.high
low=data.low
data_close_raw=data.close
data_close = data_close_raw.values
# np.flip(data.close.values, axis = 0)
volume=data.volume

close_train = data_close[:len(data_close)-PREDICT_SIZE-int(len(data_close)*VALIDATION_PCT)]
close_test = data_close[len(data_close)-PREDICT_SIZE:]
close_validate = data_close[int(len(data_close))-int(len(data_close)*VALIDATION_PCT)-PREDICT_SIZE:len(data_close)-PREDICT_SIZE]
close_validate = close_validate.reshape(-1,1)
close_test = close_test.reshape(-1, 1)
close_train = close_train.reshape(-1,1)


#Preprocessing, Scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_scaled = sc.fit_transform(close_train)
print('Scaled training Data')

x_train = []
y_train = []


for i in range(window, len(training_scaled)):
    x_train.append(training_scaled[i-window:i, 0])
    y_train.append(training_scaled[i, 0])
    print(f"""    
    x_train.append(training_scaled[{i-window}:{i}, 0]) 
    y_train.append(training_scaled[{i}, 0])""")

x_train, y_train = np.array(x_train), np.array(y_train)

#reshape     
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  


#### TESTING ####### VALIDATION #### ___ SNIPPED FROM SEGMENT AFTER MODEL AND MODIFIED

print('Scaled validation Data')
validation_scaled = sc.fit_transform(close_validate)


x_validate = []
y_validate = []

for i in range (window, len(validation_scaled)):
    x_validate.append(validation_scaled[i-window:i, 0])
    y_validate.append(validation_scaled[i, 0])
    print(i)

x_validate, y_validate = np.array(x_validate), np.array(y_validate)
x_validate = np.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1],1))

#Neural Network 
model = Sequential()

model.add(CuDNNLSTM(units = 96, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(units = 96, return_sequences = True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(units = 96, return_sequences = True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(units=96))
model.add(Dropout(0.2))

model.add(Dense(units=32))
model.add(Dense(units=1))

model.compile(optimizer = 'adam', loss= 'mean_squared_error')

# if(not os.path.exists(f'{NAME}')):
tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))
# filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
# checkpoint = ModelCheckpoint("models\{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data = (x_validate, y_validate))
model.save(f'models\\{NAME}')


model = load_model(f'models\\{NAME}')

reverse_data = data_close_raw
real_stock_price = list(reverse_data[len(reverse_data)-PREDICT_SIZE:])

dataset_total = pd.concat((reverse_data[:len(reverse_data)-PREDICT_SIZE], reverse_data[len(reverse_data)-PREDICT_SIZE:]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(close_test) - window:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
y_test = []
x_test = []

for i in range (window, int(PREDICT_SIZE+window)):
    x_test.append(inputs[i-window:i, 0])
    y_test.append(inputs[i, 0])
    # y_test.append(training_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

predicted_stock_price = model.predict(x_test) 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'red', label = f'Real {data.selection} Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = f'Predicted {data.selection} Stock Price')
plt.title(f"{data.selection} Price Prediction")
plt.xlabel('Timesteps')
plt.ylabel(f'{data.selection} Stock Price')
plt.legend()
plt.show()
