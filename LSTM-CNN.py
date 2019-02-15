import numpy as np
import os
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, TimeDistributed, Activation, Dropout, BatchNormalization
from keras.layers.recurrent import SimpleRNN, LSTM, GRU 
from keras.layers import Bidirectional
from keras import optimizers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
numberofsamples = 300
y = np.empty([numberofsamples, ], dtype=str)
# A function to get lines contents of All files in a folder
def readFile(lis, char):
    for fn in sorted(os.listdir('./data3')):  #sorted is used to sort list of files alphabetically             
          if os.path.isfile(os.path.join('./data3', fn)):
             if "py" in fn:
                 continue
             if char in fn:
                with open(os.path.join('./data3', fn)) as f:
                    content = f.readlines()
                lis.extend(content)
                f.close()
#All rows are stored sequentially in mylist
mylist = []
original_sequence_size = 4097
#readFile(mylist,'Z')  #Normal Set A
readFile(mylist,'O')  #Normal Set B
#readFile(mylist,'N')  #Inter-Ictal Set C
readFile(mylist,'F')  #Inter-Ictal Set D
readFile(mylist,'S')  #Seizure Set E
mylist = [float(i) for i in mylist]
X = np.array(mylist)
X = X.reshape(numberofsamples ,original_sequence_size)
## Z score normalization
from scipy import stats
X = stats.zscore(X, axis=1, ddof=1)
y[0:100] = "Normal"       
y[100:200] = "Inter"
y[200:300] = "Seizure"  
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
## convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
seed = 7
np.random.seed(seed)
num_units = 100
num_epochs = 50
model_batch_size =10
cmFinal = np.zeros((3,3))
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvscores = []
for train, test in kfold.split(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X[train], dummy_y[train], test_size=0.2, random_state=7)
    X_train_aug = np.repeat(X_train, 2, axis=0)    
    y_train = np.repeat(y_train, 2, axis=0)
    mask = np.random.randint(0,50,size=X_train_aug.shape).astype(np.bool)
    X_train_aug = np.multiply(X_train_aug,mask)
    from sklearn.utils import shuffle
    X_train, y_train = shuffle(X_train_aug, y_train)
    X_train = np.expand_dims(X_train  , axis=2)
    X_test =  np.expand_dims(X_test , axis=2)

# Number of hidden units to use:

    from keras import layers
    from keras.optimizers import RMSprop
    from keras.callbacks import ModelCheckpoint
    model = Sequential()
    model.add(layers.Conv1D(32, 5, activation='relu', input_shape=(4097, 1)))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling1D(3))
    model.add(layers.Conv1D(32, 5, activation='relu'))
    model.add(Bidirectional(LSTM(20, dropout=0.1, recurrent_dropout=0.5 )))
    model.add(Dense(units = 3, activation = 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)
    print(model.summary())

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=model_batch_size, callbacks=[checkpointer])
    model.load_weights('weights.hdf5')
    Y_Pred = model.predict(np.expand_dims(X[test] , axis=2))
    y_pred = np.argmax(Y_Pred, axis=1)
    y_Actual = np.argmax(dummy_y[test],axis=1)
    
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_Actual, y_pred)
    cmFinal = cmFinal + cm
    print(cm)
    print(cmFinal)
