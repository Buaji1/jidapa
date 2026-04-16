import streamlit as st
import pandas as pd
import pickle
from sklearn.datasets import load_iris

# โหลดโมเดลที่ฝึกสอนเอาไว้
with open('iris_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# โหลดข้อมูล iris อีกครั้งเพื่อนำชื่อสายพันธุ์ (target names) มาใช้แสดงผล
iris = load_iris()

st.title('🌸 Iris Flower Prediction App')
st.write("ปรับสไลเดอร์ด้านซ้ายมือเพื่อกำหนดขนาดกลีบดอกไม้ แล้วกด Predict เพื่อทำนายสายพันธุ์ของดอกไอริส")

# สร้าง Sidebar สำหรับให้ผู้ใช้ปรับค่า (เป็นเซนติเมตร)
st.sidebar.header('ปรับขนาดข้อมูล (ซม.)')

def user_input_features():
    sepal_length = st.sidebar.slider('ความยาวกลีบเลี้ยง (Sepal Length)', 4.0, 8.0, 5.4)
    sepal_width = st.sidebar.slider('ความกว้างกลีบเลี้ยง (Sepal Width)', 2.0, 5.0, 3.4)
    petal_length = st.sidebar.slider('ความยาวกลีบดอก (Petal Length)', 1.0, 7.0, 1.3)
    petal_width = st.sidebar.slider('ความกว้างกลีบดอก (Petal Width)', 0.1, 3.0, 0.2)
    
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('ข้อมูลที่คุณเลือก:')
st.write(input_df)

# เมื่อกดปุ่ม Predict
if st.button('Predict (ทำนายผล)'):
    prediction = model.predict(input_df)
    predicted_species = iris.target_names[prediction[0]]
    
    st.subheader('ผลการทำนาย:')
    st.success(f'ดอกไม้นี้คือสายพันธุ์: **{predicted_species.capitalize()}**')