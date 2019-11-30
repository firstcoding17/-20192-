from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import random as rd


RSP = pd.read_csv("data.csv")

This_Id = list(RSP["id"])
Playerlist = list(RSP["Player"])
AI = list(RSP["Lose_or_Win"])




def __main__():
    DataA = [0]*len(Playerlist)
    for i in range(0,len(Playerlist)):
        if Playerlist[i] =='rock':
            DataA[i] = 1
        elif Playerlist[i] =='paper':
            DataA[i] = 20
        elif Playerlist[i] == 'scissors':
            DataA[i] = 300
    DataX = AI
    for i in range(0,len(Playerlist)):
        DataX[i] = DataA[i] * DataX[i]
    DataY = This_Id

    X = array(DataX).reshape(len(DataX),1,1)

    # model = Sequential()
    # model.add(LSTM(50,activation='relu',input_shape=(1,1)))
    # model.add(Dense(1))
    # model.compile(optimizer='adam',loss='mse')
    # print(model.summary())
    #



    model = Sequential()
    model.add(LSTM(50,activation='relu',return_sequences=True,input_shape=(1,1)))
    model.add(LSTM(50,activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    # print(model.summary())

    model.fit(X,DataY,epochs=100,validation_split=0.01,batch_size=5)

    test_input = array([len(DataX)+1])
    test_input = test_input.reshape((1,1,1))
    test_output = model.predict(test_input,verbose=0)
    #print(test_output)

    #결과값
    output = str()

    #user => scissors
    if test_output/100 >=1:
        output = "rock"
    #user => paper
    elif test_output/100 < 1 and test_output/10 >=1:
        output = "scissors"
    #user => scissors
    elif test_output/100 < 1 and test_output/10 < 1:
        output = "paper"

    print(output)

    # X_train = DataX[:(getLen-2),:,:]
    # Y_train = DataY[:(getLen-2),:,:]
    # X_test = DataX[(getLen-2):,:,:]
    # Y_test= DataY[(getLen-2):,:,:]
    #
    # k.clear_session()
    # xInput = Input(batch_shape=(None,X_train.shape[1],X_train.shape[2]))
    # xLstm_1 = LSTM(10,return_sequences=True)(xInput)
    # xLstm_2 = Bidirectional(LSTM(10))(xLstm_1)
    # xOutput = Dense(1)(xLstm_2)
    #
    # model = Model(xInput,xOutput)
    # model.compile(loss='mse',optimizer='adam')
    #
    # model.fit(X_train,Y_train,epochs=500,batch_size=20,verbose=1)
    #
    # y_hat = model.predict(X_test,batch_size=1)
    #
    # a_axis = np.arange(0,len(Y_train))
    # b_axis = np.arange(len(Y_train),len(Y_train)+len(y_hat))
    # plt.figure(figsize=(10,6))
    # plt.plot(a_axis,Y_train.reshape[70,],'o-')
    # plt.plot(b_axis,y_hat.reshape(20,),'o-',color='red',label='Predicted')
    # plt.plot(b_axis,Y_test.reshape(20,),'o-',color='green',alpha=0.2,label='Actual')
    # plt.legend()
    # plt.show()



    # model = Sequential()
    # model.add(Dense(32,input_dim=40,activation="relu"))
    # model.add(Dropout(0,3))
    # for i in range(2):
    #     model.add(Dense(32,activation="relu"))
    #     model.add(Dropout(0,3))
    # model.add(Dense(1))
    # 다증 퍼셉트론 모델
    #

    # model = Sequential()
    # model.add(LSTM(32,input_shape=(None,1)))
    # model.add(Dropout(0,3))
    # model.add(Dense(1))
    #순환신경망 모델

    # model = Sequential()
    # model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
    # model.add(Dropout(0.3))
    # model.add(Dense(1))
    #상태 유지 신경망



if __name__ == "__main__":
    __main__()