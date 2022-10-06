#Imports
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score

#Load Training set
xtrain_tacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\Inertial Signals\total_acc_x_train.txt',header = None, delim_whitespace=True))
ytrain_tacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\Inertial Signals\total_acc_y_train.txt',header = None, delim_whitespace=True))
ztrain_tacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\Inertial Signals\total_acc_z_train.txt',header = None, delim_whitespace=True))
xtrain_bacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\Inertial Signals\body_acc_x_train.txt',header = None, delim_whitespace=True))
ytrain_bacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\Inertial Signals\body_acc_y_train.txt',header = None, delim_whitespace=True))
ztrain_bacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\Inertial Signals\body_acc_z_train.txt',header = None, delim_whitespace=True))
xtrain_gyro = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\Inertial Signals\body_gyro_x_train.txt',header = None, delim_whitespace=True))
ytrain_gyro = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\Inertial Signals\body_gyro_y_train.txt',header = None, delim_whitespace=True))
ztrain_gyro = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\Inertial Signals\body_gyro_z_train.txt',header = None, delim_whitespace=True))

y_train = pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\train\y_train.txt',header = None, delim_whitespace=True)


#Load test data
xtest_tacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\total_acc_x_test.txt', header = None, delim_whitespace=True))
ytest_tacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\total_acc_y_test.txt', header = None, delim_whitespace=True))
ztest_tacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\total_acc_z_test.txt', header = None, delim_whitespace=True))
xtest_bacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_acc_x_test.txt', header = None, delim_whitespace=True))
ytest_bacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_acc_y_test.txt', header = None, delim_whitespace=True))
ztest_bacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_acc_z_test.txt', header = None, delim_whitespace=True))
xtest_gyro = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_gyro_x_test.txt', header = None, delim_whitespace=True))
ytest_gyro = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_gyro_y_test.txt', header = None, delim_whitespace=True))
ztest_gyro = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_gyro_z_test.txt', header = None, delim_whitespace=True))

#NORMALISING
#Test Labels
y_test = pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\y_test.txt',header = None)

def scaling(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train.reshape(-1,1))
    trainN = scaler.transform(train.reshape(-1,1)).reshape(train.shape)
    testN = scaler.transform(test.reshape(-1,1)).reshape(test.shape)
    return trainN, testN

#Signal scaling
xtrain_taccN, xtest_taccN = scaling(xtrain_tacc, xtest_tacc)
xtrain_baccN, xtest_baccN = scaling(xtrain_bacc, xtest_bacc)
xtrain_gyroN, xtest_gyroN = scaling(xtrain_gyro, xtest_gyro)
ytrain_taccN, ytest_taccN = scaling(ytrain_tacc, ytest_tacc)
ytrain_baccN, ytest_baccN = scaling(ytrain_bacc, ytest_bacc)
ytrain_gyroN, ytest_gyroN = scaling(ytrain_gyro, ytest_gyro)
ztrain_taccN, ztest_taccN = scaling(ztrain_tacc, ztest_tacc)
ztrain_baccN, ztest_baccN = scaling(ztrain_bacc, ztest_bacc)
ztrain_gyroN, ztest_gyroN = scaling(ztrain_gyro, ztest_gyro)


#Combine 9 channels together 
x_train = [xtrain_taccN, ytrain_taccN, ztrain_taccN,
            xtrain_baccN, ytrain_baccN, ztrain_baccN,
           xtrain_gyroN, ytrain_gyroN, ztrain_gyroN]
x_test = [xtest_taccN, ytest_taccN, ztest_taccN,
          xtest_baccN, ytest_baccN, ztest_baccN,
          xtest_gyroN, ytest_gyroN, ztest_gyroN]

x_train = np.array(np.dstack(x_train),dtype=np.float32)
x_test = np.array(np.dstack(x_test),dtype = np.float32)


#make the label's index zero, necessary as one-hot encoding starts from 0 and not 1
y_train = y_train-1
y_test = y_test-1

#One-hot encoding
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
#y_test changed to a data array here, must be for total accuracy check

#Build CNN Model

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 4), activation='relu', padding='same',input_shape=(128,9,1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding ='same'))
model.add(tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 4), activation='relu', padding='same',input_shape=(128,9,1)))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding ='same'))
#Number of layers changed here. Currently set to two

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))
model.add(tf.keras.layers.Dense(6, activation='softmax'))

#Optimisation
model.compile(optimizer = tf.keras.optimizers.Adam(0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#Train Model
x_train=x_train.reshape(7352,128,9,1)
history = model.fit(x_train,y_train, epochs = 30, batch_size = 64,
          validation_split = 0.2, shuffle = True)
x_test=x_test.reshape(2947,128,9,1)


#Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, batch_size = 64)

model.summary()

#Save model
name = input("Save model name: ")
modelpath='saved_models/'+name+'_model'
model.save(modelpath)


#Obtain predictions
predictions = model.predict(x_test)
predictions = np.around(predictions, decimals=0, out=None)

#F1 Score
F1 = f1_score(y_test, predictions, average='micro')
print('F1 score =', F1)

#Accuracy calculation
loop = 0
accuracy = 0
for loop in range (0, 2947 ):
    if(np.argmax(predictions[loop]) == np.argmax(y_test[loop])):
        accuracy+=1

print("Model accuracy =", accuracy/29.47) 

#Converts to a TensorFlow Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]

#Converting model
tflite_model = converter.convert()
open(modelpath + '.tflite', 'wb').write(tflite_model)

#TFlite Interpreter for Inference
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()  #Gets model input tensor details


# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


#Run Interpreter in for loop for every test data entry 
acc = 0
input_shape = input_details[0]['shape']

for i in range(0,len(x_test)): #Changed from 2947 - these are the test cases
    interpreter.set_tensor(input_details[0]['index'], x_test[i:i+1,:,:]) #Sets value of input tensor
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if(np.argmax(output_data) == np.argmax(y_test[i])):
        acc+=1
acc = 100*acc/len(x_test) #Calcuates accuracy as a percentage
print("\nTFLite percentage model accuracy:")
print(acc)