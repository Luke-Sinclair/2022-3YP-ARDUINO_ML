#For sending test data over the serial port
#Imports
import numpy as np
import struct
import time
import pandas as pd
import serial
from sklearn.preprocessing import MinMaxScaler

#Load Training Set
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


#Load Testing Set
xtest_tacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\total_acc_x_test.txt', header = None, delim_whitespace=True))
ytest_tacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\total_acc_y_test.txt', header = None, delim_whitespace=True))
ztest_tacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\total_acc_z_test.txt', header = None, delim_whitespace=True))
xtest_bacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_acc_x_test.txt', header = None, delim_whitespace=True))
ytest_bacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_acc_y_test.txt', header = None, delim_whitespace=True))
ztest_bacc = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_acc_z_test.txt', header = None, delim_whitespace=True))
xtest_gyro = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_gyro_x_test.txt', header = None, delim_whitespace=True))
ytest_gyro = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_gyro_y_test.txt', header = None, delim_whitespace=True))
ztest_gyro = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\Inertial Signals\body_gyro_z_test.txt', header = None, delim_whitespace=True))


y_test = np.array(pd.read_csv(r'C:\Users\lukes\.spyder-py3\HAR Dataset\test\y_test.txt',header = None, delim_whitespace=True))

def scaling(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train.reshape(-1,1))
    trainN = scaler.transform(train.reshape(-1,1)).reshape(train.shape)
    testN = scaler.transform(test.reshape(-1,1)).reshape(test.shape)
    return trainN, testN

#Scale signals
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

x_test = [xtest_taccN, ytest_taccN, ztest_taccN,
          xtest_baccN, ytest_baccN, ztest_baccN,
          xtest_gyroN, ytest_gyroN, ztest_gyroN]

x_test = np.array(np.dstack(x_test),dtype = np.float32)

y_train = y_train-1
y_test = y_test-1

loops = 2947 #Number of test data classification windows
acc = 0 #Accuracy counter
total_acc = 0 #Inference accuracy
a=0 #Prediction variable
t=0 #Latency variable
Inf = 0 #Latency accumulation
avgInf = 0

for i in range (0, int(loops)):
    #Sending x_test over Serial
    ser = serial.Serial('COM3', 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE, timeout=10.0)
    #should be COM3 as we are sending data to the Arduino
    print("Sending data")
    for j in range (0,128):
        for k in range (0, 9):
            s_input = x_test[int(i)][j][k]
            if ser.out_waiting < 33:#as another 32 bytes would push past 64
                bin = struct.pack('f',s_input)  #stores float as binary bytes
                ser.write(bin) #Writes to serial port
            else:
                time.sleep(0.001)
                print("Output maxed")
        
    while ser.inWaiting() == 0 : #Waits for Arduino inference prediction and latency
        continue
    
    a=int(ser.readline().strip()) #read prediction
    t=int(ser.readline().strip()) #read latency
    print("Prediction", i, "read from serial port:", a) 
    print("Inference time:", t)

    Inf += t #Accumulation
    if y_test[i][0] == a: #if prediction true
        acc = acc+1 #increment accuracy counter
    
    ser.close() #Serial port temporarily closed to reset buffers
    continue #for loop repeats for next classification window

total_acc = 100*(acc / loops) #Arduino inference accuracy calculation
print("Total accuracy is: ", total_acc, "%")
avgInf = Inf / loops #Average latency calculation
print("The average inference time is:", avgInf)
print("Program ended")
