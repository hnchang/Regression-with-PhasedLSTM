import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
from keras.models import load_model
from keras import models 
from keras import layers
from PhasedLSTM import PhasedLSTM as PLSTM
from keras.callbacks import Callback
import seaborn as sns
import pandas as pd
import sys, time
import numpy as np
import warnings


# source: https://fairyonice.github.io/Extract-weights-from-Keras's-LSTM-and-calcualte-hidden-and-cell-states.html

def set_seed(sd=123):
    from numpy.random import seed
    from tensorflow import set_random_seed
    import random as rn
    ## numpy random seed
    seed(sd)
    ## core python's random number 
    rn.seed(sd)
    ## tensor flow's random number
    set_random_seed(sd)
    
def random_sample(len_ts=3000,D=1001):
    c_range = range(5,100)
    c1 = np.random.choice(c_range)
    u = np.random.random(1)
    const = -1.0/len_ts
    ts = np.arange(0,len_ts)
    
    omega_c = 1.0/float(1.0 + c1)
    # omega_m = 2*omega_c
    
    x1 = np.cos(ts*omega_c)
    x1 = x1*ts*u*const
    # x2 = np.cos(ts*omega_c + np.cos(ts*omega_m))
    
    x2 = np.cos(2*ts*omega_c)
    
    y1 = np.zeros(len_ts)

    for t in range(D,len_ts):
        ## the output time series depend on input as follows: 
        # y1[t] = x1[t-2]*x1[t-D]
        y1[t] = x1[t-2]*x2[t-D] 
    y = np.array([y1]).T
    X = np.array([x1]).T
    return y, X


def generate_data(D= 1001,Nsequence = 1000,T=4000, seed=123):
    X_train = []
    y_train = []
    set_seed(sd=seed)
    for isequence in range(Nsequence):
        y, X = random_sample(T,D=D)
        X_train.append(X)
        y_train.append(y)
    return np.array(X_train),np.array(y_train)

D = 10
T = 1000
X, y = generate_data(D=D,T=T,Nsequence = 1000)
print(X.shape, y.shape)

    
def plot_examples(X,y,ypreds=None,nm_ypreds=None):
    fig = plt.figure(figsize=(16,10))
    fig.subplots_adjust(hspace = 0.32,wspace = 0.15)
    count = 1
    n_ts = 16
    for irow in range(n_ts):
        ax = fig.add_subplot(n_ts/4,4,count)
        ax.set_ylim(-0.5,0.5)
        ax.plot(X[irow,:,0],"--",label="x1")
        ax.plot(y[irow,:,:],label="y",linewidth=3,alpha = 0.5)
        ax.set_title("{:}th time series sample".format(irow))
        if ypreds is not None:
            for ypred,nm in zip(ypreds,nm_ypreds):
                ax.plot(ypred[irow,:,:],label=nm)   
        count += 1
    plt.legend()
    plt.show()
plot_examples(X,y,ypreds=None,nm_ypreds=None)


class ModelHistory(Callback):
    def __init__(self):
        self.batch = 0

    # Save the model after every batch in the training set to check the kernel weights.
    def on_batch_end(self, batch, logs=None):
        name = 'model%03d.hdf5' % self.batch
        self.model.save(name)
        self.batch += 1

def define_model(len_ts,
                 latent_neurons = 1,
                 nfeature=1,
                 batch_size=None,
                 stateful=False):
    in_out_neurons = 1
    
    inp = layers.Input(batch_shape= (batch_size, len_ts, nfeature),
                       name="input")  

#    rnn = layers.LSTM(latent_neurons, 
#                    return_sequences=True,
#                    stateful=stateful,
#                    name="RNN")(inp)

# Compare the PLSTM with LSTM.
    rnn = PLSTM(latent_neurons, 
                    return_sequences=True,
                    stateful=stateful,
                    name="RNN")(inp)
                    
    dens = layers.Dense(units = in_out_neurons, name="dense", activation='tanh')(rnn)
    model = models.Model(inputs=[inp],outputs=[dens])

    model.compile(loss="mean_squared_error",
                  sample_weight_mode="temporal",
                  optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0))
                  
    return(model,(inp,rnn, dens))


X_train, y_train = X ,y
hunits = 10
model1, _ = define_model(latent_neurons = hunits, len_ts = X_train.shape[1])
model1.summary()

w = np.zeros(y_train.shape[:2])
w[:,D:] = 1 
w_train = w

start = time.time()
hist1 = model1.fit(X_train, y_train, 
                   batch_size=2**5,
                   epochs=200, 
                   verbose=1,
                   sample_weight=w_train,
                   validation_split=0.05,
                   callbacks=[ModelHistory()])
end = time.time()
print("Time took {:3.1f} min".format((end-start)/60))


labels = ["loss","val_loss"]
for lab in labels:
    plt.plot(hist1.history[lab],label=lab + " model1")
plt.yscale("log")
plt.legend()
plt.show()



# Validate the model performance with new data

X_test, y_test = generate_data(D=D,T=T,seed=2, Nsequence = 1000)
y_pred1 = model1.predict(X_test)

w_test = np.zeros(y_test.shape[:2])
w_test[:,D:] = 1

plot_examples(X_test,y_test,ypreds=[y_pred1],nm_ypreds=["ypred model1"])
print("The final validation loss is {:5.4f}".format( 
    np.mean((y_pred1[w_test == 1] - y_test[w_test==1])**2 )))


model_path = 'model003.hdf5' # Hard-code here to correspond to the filename in ModalHistory callback.
h5_model = load_model(model_path, custom_objects={'PhasedLSTM': PLSTM} )


# for plstm (phased LSTM)
#        period = self.timegate_kernel[0]
#        shift = self.timegate_kernel[1]
#        r_on = self.timegate_kernel[2]
#

for layer in h5_model.layers:
        if "PhasedLSTM" in str(layer):
            weightPLSTM = layer.get_weights()
            print (weightPLSTM)

