import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# import googleapiclient.discovery # PDF 코드에는 있지만 실제 앱 로직에서는 사용되지 않음
import os
from flask import Flask, render_template, request # request 추가
# from dotenv import load_dotenv # PDF 코드에는 있지만 실제 앱 로직에서는 사용되지 않음
from tensorflow import keras # 모델 로드를 위해 추가

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

from flask_bootstrap import Bootstrap5
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

bootstrap5 = Bootstrap5(app)

# Form Class Definition
class LabForm(FlaskForm):
    preg = StringField('# Pregnancies', validators=[DataRequired()])
    glucose = StringField('Glucose', validators=[DataRequired()])
    blood = StringField('Blood pressure', validators=[DataRequired()])
    skin = StringField('Skin thickness', validators=[DataRequired()])
    insulin = StringField('Insulin', validators=[DataRequired()])
    bmi = StringField('BMI', validators=[DataRequired()])
    dpf = StringField('DPF Score', validators=[DataRequired()])
    age = StringField('Age', validators=[DataRequired()])
    submit = SubmitField('Submit')

# ---- 스케일러 및 모델 로드 (앱 시작 시 한 번만 수행하도록 수정 권장) ----
# 원본 데이터 로드 (스케일러 학습용 - 비효율적이지만 PDF 코드대로)
data = pd.read_csv('./diabetes.csv', sep=',') # diabetes.csv 파일이 같은 폴더에 있어야 함
X_for_scaler = data.values[:, 0:8]

# MinMaxScaler 생성 및 학습
scaler = MinMaxScaler()
scaler.fit(X_for_scaler)

# 모델 로드
model = keras.models.load_model('pima_model.keras') # pima_model.keras 파일이 같은 폴더에 있어야 함
# -------------------------------------------------------------------

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['GET', 'POST'])
def lab():
    form = LabForm()
    if form.validate_on_submit():
        # get the form data for the patient data and put into a form for the model
        X_test = np.array([[float(form.preg.data),
                            float(form.glucose.data),
                            float(form.blood.data),
                            float(form.skin.data),
                            float(form.insulin.data),
                            float(form.bmi.data),
                            float(form.dpf.data),
                            float(form.age.data)]])

        print("Original Input Shape:", X_test.shape)
        print("Original Input Data:", X_test)

        # min max scale the data for the prediction (미리 로드된 scaler 사용)
        X_test_scaled = scaler.transform(X_test)
        print("Scaled Input Data:", X_test_scaled)

        # evaluate model (미리 로드된 model 사용)
        prediction = model.predict(X_test_scaled)
        res_raw = prediction[0][0] # 예측 결과 (확률)

        # 결과를 백분율로 변환
        res_percent = np.round(res_raw * 100, 2) # 소수점 둘째 자리까지

        print("Raw Prediction:", res_raw)
        print("Prediction Percent:", res_percent)

        return render_template('result.html', res=res_percent) # 결과 페이지로 전달

    # GET 요청 시 폼 렌더링
    return render_template('prediction.html', form=form)

if __name__ == '__main__':
    app.run(debug=True) # 개발 시 debug 모드 활성화