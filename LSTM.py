#Project
#Data-Driven Methods HT 2022
#Carlos Neves - LSTM
#%%
from tkinter import X
import Project_CarlosNeves as flow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


#%%

u, v, u_mean, v_mean, u_fluc, v_fluc, vorticity, velocity, velocity_mean, velocity_fluc, x_coord, y_coord, x_vort_coord, y_vort_coord, dx_vort, dy_vort, fignum=flow.Flow(min_t=999, max_t=999, t_frames=1)
#%%
#------Start-----
#Creates the data matrix 
X_u=np.zeros((8192,1000)) 
X_v=np.zeros((8192,1000))
X_velocity=np.zeros((8192,1000))
X_vorticity=np.zeros((7812,1000))

#--Preprocessing data
#%%
#Reshaping the data as matrices with rows corresponding to velocity component info accross
#the domain and with columns corresponding to time snapshots
for t in list(range(1000)):
    uu=np.copy(u[t,:,:]) #takes the u component velocity at the time step t
    uu_r=np.reshape(np.transpose(uu),(8192)) #reshapes the nxm grid into a vector - transpose is implemented so the velocity at a certain x is turned into the rows that are later reshaped into columns
    X_u[:,t]=np.copy(uu_r) #makes the column of the data matrix equal to the previous vector 
    
    vv=np.copy(v[t,:,:])
    vv_r=np.reshape(np.transpose(vv),(8192))
    X_v[:,t]=np.copy(vv_r)
#%%
X_u_mean = np.mean(X_u,axis=-1)
X_u_std = np.std(X_u,-1)

#%%
## u_data.shape = 1000,
LEN = X_u.shape[0]
for i in range(LEN):
    X_u[i,:] = (X_u[i,:] - X_u_mean[i])/X_u_std[i]
#%%
input_size=10 #INPUT - number of time steps used in each batch
X_train = np.empty(shape=(LEN-1,1000-input_size,input_size,1))
Y_train = np.empty(shape=(LEN-1,1000-input_size))

for series in range(LEN-1):
    u_data=np.copy(X_u[series,:])

    u_train_set=u_data #Define the size of the training data set
    # u_test_set=u_data[np.shape(u_train_set)[0]:]  #Test data set

    

    x_train=np.zeros([np.shape(u_train_set)[0]-input_size, input_size,1])
    y_train=np.zeros(np.shape(u_train_set)[0]-input_size)

    for i in range(input_size,np.shape(u_train_set)[0]):
        x_train[i-input_size,:,0]=u_train_set[i-input_size:i]
        y_train[i-input_size]=u_train_set[i]

    X_train[series,:,:] = x_train
    Y_train[series,:] = y_train
#%%
tf.keras.backend.clear_session()
#Initialising the RNN  
regressor=Sequential()

#Adding the first LSTM Layer and some Dropout regularization
#regressor.add(LSTM(units=10, return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(LSTM(units=90, input_shape=(x_train.shape[1],1),return_sequences=True,name = "LSTM1"))
regressor.add(LSTM(units=90, return_sequences=False,name = "LSTM2"))
regressor.add(Dense(units=1,name = "MLP1"))
LR = 0.001
STEP_EPOCHS = 20
N_EPOCHS = 100
BATCH_SIZE=32
n_train = int(10**np.floor(np.log10(x_train.shape[0]))/2)

steps_per_epoch = int(np.ceil(n_train/BATCH_SIZE))
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            LR,
            decay_steps=int(steps_per_epoch*STEP_EPOCHS),
            decay_rate=0.001,
            staircase=True)

optimizer_A = tf.keras.optimizers.Adam(learning_rate=LR)

regressor.compile(optimizer=optimizer_A, loss = 'mean_squared_error')
print(regressor.summary())
#%%
N_series = 40
all_history = []

for series in range(N_series):
    print(f"At {series} Series ")
    x_train= X_train[series,:,:,:]
    y_train = Y_train[series,:]
    y_train = np.expand_dims(y_train,-1)
    history =regressor.fit(x_train, y_train, 
                            epochs=N_EPOCHS, 
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            verbose =0)
    hist = history.history

    all_history.append(hist["loss"])
#%%
regressor.save("LSTM_2.h5")
#%%
loss = np.empty(N_series*N_EPOCHS)
for i in range(len(all_history)):
    loss[i*100:100*(i+1)] = np.array(all_history[i])
plt.plot(loss)
# plt.plot(hist["val_loss"])
plt.show()

#%%
#Predicting the data
u_test_set = X_u[-1]

x_test=np.zeros([np.shape(u_test_set)[0]-input_size, input_size,1])

# y_test=np.zeros(np.shape(u_test_set)[0]-input_size)

for i in range(input_size,np.shape(u_test_set)[0]):
        x_test[i-input_size,:,0]=u_test_set[i-input_size:i]
        # y_test[i-input_size]=u_test_set[i]

predicted_data = regressor.predict(x_test)
#%%
plt.figure()
plt.plot(predicted_data,label="prediction")
plt.plot(u_test_set,label="org")
# plt.plot(u_data,label="org")
plt.legend()
plt.show()
# %%
