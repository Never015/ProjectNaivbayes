import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# 📌 โหลดข้อมูล Heart Dataset
Heart = pd.read_csv("./data/heart02.csv")
X = Heart.drop(columns=['HeartDisease'])
y = Heart.HeartDisease
HeartDisease=Heart.HeartDisease


# 📌 สร้างและฝึกโมเดล Naïve Bayes
model = GaussianNB()
model.fit(X, y)

# 📌 สร้าง Web App ด้วย Streamlit
st.title("HeartDisease - Naïve Bayes")
st.write("กรอกข้อมูลHeartDisease แล้วให้โมเดลทำนายHeartDisease")

# 📌 สร้างอินพุตให้ผู้ใช้กรอกข้อมูล
A1 = st.number_input("กรุณาเลือกข้อมูล Age(อายุ)")
A2 = st.selectbox("กรุณาเลือก Sex(เพศ)ชาย=1 หญิง=0",[0,1])
A3 = st.selectbox("กรุณาเลือก ChestPainType ASY = 1 ATA =2 NAP = 3 TA = 4",[1,2,3,4])
A4 = st.sidebar("กรุณาเลือกข้อมูล RestingBP 0 - 200")
A5 = st.sidebar("กรุณาเลือกข้อมูล Cholesterol 0 - 603")
A6 = st.selectbox("กรุณาเลือก FastingBS ",[0,1])
A7 = st.selectbox("กรุณาเลือก RestingECG LVH = 1 Normal = 2 ST =3",[1,2,3])
A8 = st.sidebar("กรุณาเลือกข้อมูล MaxHR 0 - 202")
A9 = st.selectbox("กรุณาเลือก ExerciseAngina Y = 1 N = 0",[0,1])
A10 = st.sidebar("กรุณาเลือกข้อมูล Oldpeak -2.6 - 6.2")
A11= st.selectbox("กรุณาเลือก ST_Slope Down = 1 Flat = 2 UP = 3",[1,2,3])

# 📌 ปุ่มกดเพื่อทำนาย
if st.button("🔍 Predict"):
    user_input = np.array([[A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11]])
    prediction = model.predict(user_input)
    predicted_class = HeartDisease[prediction[0]]
    
    st.success(f"🌼 ผลลัพธ์: ขอการเป็นโรคหัวใจ **{predicted_class}**")

