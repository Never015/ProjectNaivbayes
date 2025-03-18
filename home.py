import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

# 📌 โหลดข้อมูล Iris Dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 📌 สร้างและฝึกโมเดล Naïve Bayes
model = GaussianNB()
model.fit(X, y)

# 📌 สร้าง Web App ด้วย Streamlit
st.title("🌸 Iris Flower Classifier - Naïve Bayes")
st.write("กรอกข้อมูลของดอกไอริส แล้วให้โมเดลทำนายชนิดของดอกไม้")

# 📌 สร้างอินพุตให้ผู้ใช้กรอกข้อมูล
sepal_length = st.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width (cm)", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width (cm)", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# 📌 ปุ่มกดเพื่อทำนาย
if st.button("🔍 Predict"):
    user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(user_input)
    predicted_class = target_names[prediction[0]]
    
    st.success(f"🌼 ผลลัพธ์: ดอกไอริสชนิด **{predicted_class}**")

