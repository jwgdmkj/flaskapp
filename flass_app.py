#pip3 install -r requirements.txt 을 통해,
#requirements.txt에 numpy==1.1.4 pandas==1.0.3 등등을 적어두면
#한 번에 패키지 다운이 가능
#쉘 스크립트로, install.sh 파일에 pip3 install -r requirements.txt를 적어두고
# chmod -x install.sh && ./install.sh 실행시, 패키지 다운 가능

#flask-app 파일을 zip으로 압축하면, 어디든 install.sh와 run_flask_app.sh를 실행한다면, 서버가 실행됨
#백그라운드 실행법 - nohup python3 flask-app.py &
#로그 보는법 - tail -f nohup.out

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler

from flask import Flask, render_template, request
#from flask_ngrok import run_with_ngrok

#load data
X = pd.read_csv('model/X.csv')

with open('model/y.npy', 'rb') as f:
  y = np.load(f)

X = X[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', 
       '1stFlrSF', 'FullBath', 'LotShape_rank']]


x_min_max_scaler = MinMaxScaler()
y_min_max_scaler = MinMaxScaler()
x_min_max_scaler.fit(X)
y_min_max_scaler.fit(y) #X, y의 feature별 최대최소값을 스케일러가 기억

scaled_X = x_min_max_scaler.transform(X)
scaled_y = y_min_max_scaler.transform(y)

model = keras.Sequential(
      [
          keras.Input(shape=scaled_X.shape[-1]),
          layers.Dense(96, activation='relu'),
          layers.Dense(48, activation='relu'),
          layers.Dense(1)
      ]
  )

model.compile(loss="mse", optimizer="adam")

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
model.fit(scaled_X, scaled_y, 
          batch_size=2, epochs=150, 
          callbacks=[early_stopping_callback], validation_split=0.05)

pred = model.predict(scaled_X[:1]) # 0 ~ 1
pred = y_min_max_scaler.inverse_transform(pred)

#앞서 학습시킨 모델(adam형 compile)을, 학습시키지 않고, 바로 모델을 저장하고 load해서 쓰고자할때
#학습된 모델을, h5라는 형태로 저장하고, load_model 통해 그대로 쓸 수 있음
#model.save("mlp_v0.1.h5")

#load model
reconstructed_model = keras.models.load_model("model/mlp_v0.1.h5")

pred = reconstructed_model.predict(scaled_X[:1]) # 0 ~ 1
pred = y_min_max_scaler.inverse_transform(pred)

#pred

#submit form이 위치한 곳을 알려줌(/content)
app = Flask(__name__)
#run_with_ngrok(app) #flask 앱 선언 후, copy 할 수 있도록 함(ngrok에 권한 줌)

def preprocess_data(data) :
  #TODO: 전처리. 8자리 정보를 np 딕셔너리 array로 변경시킨다(1,8짜리)
  #이 때, LotShpae는 str로 들어오므로, 이를 적절한 값으로 변환이 필요
  X = [] # <-- 각각 값을 넣을 배열
  for k, v in data.items() :
    if k == 'LotShape' :
      if v=='Reg' :
        X.append(4)
      elif v == 'IR1' :
        X.append(3)
      elif v == 'IR2':
        X.append(2)
      else :
        X.append(1)
    else :
      X.append(float(v))

  # X = [2, 5000, 2, ..., 3]
  X = np.array(X) #(8,)
  X = X.reshape((1, -1)) #(1,8)

  scaled_X = x_min_max_scaler.transform(X)
  print(scaled_X.shape)
  return scaled_X

@app.route("/")
def predict() :
  #return "<h1> This is your flask server</h1>"
  return render_template("submit_form.html")

#submit버튼 누르면 result가 호출됨(form action = "/result"에 의해)
@app.route("/result", methods=['POST'])
def result():
  #data 읽고, data전처리 후, Model prediction시켜 값 리턴
  #Salpeprice와 가장 상관관계 높은 10개의 것을 선택해 집값 유추하고자함
  #GrLivArea의 경우, X['GrLivArea'].min(), max(), median()을 통해 값 범위 보자

  #받아온 데이터를 읽어들이자

  data = request.form

  message = ""
  message +="<hl> House Price </hl>"

  for k, v in data.items():
    print(k,v)
    message += k+": " + v + "</br>" 
  
  #데이터 전처리
  X = preprocess_data(data) #user가 보낸 data를
  # X: (1,8)
  
  #pred = model.predict(X)
  pred = reconstructed_model.predict(X)
  pred = y_min_max_scaler.inverse_transform(pred)
  #이 때, pred shape는 (1,1).

  message += "</br>"
  message += "Predict price: " + str(pred[0][0]) 
  print(message)
  return message

app.run() #앱실행. 유저가 요청 보낼 함수를 여러 개 만들어두면, 더 복잡한 앱서버 생성이 가능
#지금까진, 코랩 터미널에서 요청 보내면, 내가 응답을 받게 됨. 
#ngrok에 대신 요청을 보내면, 코랩을 찾아서 응답을 카피해 나한테 보내줄 것. 이 떄, 난 외부사용자(edge)

#