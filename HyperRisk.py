import streamlit as st
import pandas as pd
import numpy as np
import joblib
import random

# โหลดโมเดลและ scaler
model = joblib.load("hypertension_model.pkl")
scaler = joblib.load("scaler.pkl")

# ตั้งชื่อหน้าเว็บ
st.title("Hypertension Risk Predictor without Blood Pressure and Heart Rate")
st.write("ระบบทำนายความเสี่ยงความดันโลหิตสูงจากปัจจัยสุขภาพส่วนบุคคล โดยไม่ใช้ค่าความดันหรือค่าชีพจร")

# ฟังก์ชันสร้าง user input
def get_user_input():
    Age = st.number_input("อายุ", 18, 100, 30)
    Sex = st.selectbox("เพศ", ["หญิง", "ชาย"])
    BMI = st.number_input("BMI", 15.0, 40.0, 24.0)
    Sleep = st.number_input("ชั่วโมงการนอน", 3.0, 10.0, 7.0)
    Stress = st.slider("ระดับความเครียด (1-10)", 1, 10, 5)
    Smoke = st.selectbox("สูบบุหรี่หรือไม่", ["ไม่", "ใช่"])
    Alcohol = st.selectbox("ดื่มแอลกอฮอล์หรือไม่", ["ไม่", "ใช่"])
    Exercise = st.selectbox("ออกกำลังกายหรือไม่", ["ไม่", "ใช่"])
    Disease = st.selectbox("โรคประจำตัว", ["ไม่มี", "เบาหวาน", "โรคหัวใจ", "ไตเรื้อรัง"])
    
    # Mapping
    sex_val = 0 if Sex == "หญิง" else 1
    smoke_val = 0 if Smoke == "ไม่" else 1
    alcohol_val = 0 if Alcohol == "ไม่" else 1
    exercise_val = 0 if Exercise == "ไม่" else 1
    disease_val = {"ไม่มี":0, "เบาหวาน":1, "โรคหัวใจ":2, "ไตเรื้อรัง":3}[Disease]
    
    data = pd.DataFrame({
        "Age": [Age],
        "Sex": [sex_val],
        "BMI": [BMI],
        "Sleep": [Sleep],
        "Stress": [Stress],
        "Smoke": [smoke_val],
        "Alcohol": [alcohol_val],
        "Exercise": [exercise_val],
        "Disease": [disease_val]
    })
    
    # Interaction features
    data["BMIxStress"] = data["BMI"] * data["Stress"]
    data["AgeOverBMI"] = data["Age"] / data["BMI"]
    data["SleepPerStress"] = data["Sleep"] / (data["Stress"] + 1)
    
    return data

user_data = get_user_input()

# คำแนะนำ
low_msgs = ["สุขภาพโดยรวมดีมาก", "อยู่ในเกณฑ์ปลอดภัย", "ยังไม่พบความเสี่ยงสำคัญในตอนนี้"]
low_advice = ["รักษาพฤติกรรมที่ดีไว้", "พักผ่อนให้เพียงพอ ออกกำลังกาย 3 ครั้งต่อสัปดาห์ เพื่อป้องกันความดันในอนาคต"]

moderate_msgs = ["เริ่มมีแนวโน้มความเสี่ยง", "สุขภาพเริ่มอ่อนแอ เข้าข่ายมีความเสี่ยง"]
moderate_advice = ["ปรับเปลี่ยนพฤติกรรมในการใช้ชีวิตบางอย่างที่สามารถส่งผลกระทบได้ เช่น การนอน หรือ การสูบ หรือ การดื่ม", "สังเกตพฤติกรรมที่มีโอกาสเข้าข่ายในความเสี่ยงนี้ เช่น ความเครียดและการนอน"]

high_msgs = ["พบความเสี่ยงสูง", "ค่าความเสี่ยงของคุณสูงกว่าค่าเฉลี่ยมาก", "ร่างกายมีสัญญาณเตือนถึงความเสี่ยง"]
high_advice = ["ควรปรึกษาแพทย์เพิ่มเติมเพื่อตรวจความดันโลหิตโดยละเอียด"]

import matplotlib.pyplot as plt

# ปุ่มทำนาย
if st.button("วิเคราะห์ความเสี่ยง"):
    # Scale
    user_scaled = scaler.transform(user_data)

    # Predict
    prediction = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0]

    # แสดงผล
    if prediction == 0:
        st.success("🟢 ความเสี่ยงต่ำ (Low Risk)")
        st.write(random.choice(low_msgs))
        st.info(random.choice(low_advice))
    elif prediction == 1:
        st.warning("🟡 ความเสี่ยงปานกลาง (Moderate Risk)")
        st.write(random.choice(moderate_msgs))
        st.info(random.choice(moderate_advice))
    else:
        st.error("🔴 ความเสี่ยงสูง (High Risk)")
        st.write(random.choice(high_msgs))
        st.info(random.choice(high_advice))

    st.write(f"ค่าความมั่นใจของโมเดล: {np.max(prob):.2%}")

    labels = ["Low", "Moderate", "High"]
    colors = ["#8FCB9B", "#FFD166", "#D72638"] 
    fig, ax = plt.subplots()
    ax.pie(prob, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors, textprops={'fontsize': 12})
    ax.axis("equal")
    st.pyplot(fig)

# ปรับโทนสีเว็บ (ใช้ CSS ตกแต่ง)
st.markdown("""
    <style>
    /* พื้นหลังหลัก */
    .stApp {
        background-color: #1E1A1D; /* ดำอมแดงเข้ม */
        color: #F4F1F0; /* ขาวนวล */
        font-family: 'Prompt', sans-serif;
    }

    /* ปุ่ม */
    div.stButton > button {
        background-color: #D72638; /* แดงเชอร์รี่ */
        color: #FFF;
        border-radius: 10px;
        height: 3em;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    div.stButton > button:hover {
        background-color: #A4161A;
        color: #FFF;
    }

    /* กล่องข้อความและ input box */
    .stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stTextArea {
        background-color: #2E2A2D !important; /* เทาเข้มอมม่วง */
        border-radius: 8px;
        color: #F4F1F0 !important;
    }

    /* หัวข้อ */
    h1, h2, h3 {
        color: #FF6B6B; /* ชมพูเชอร์รี่ */
    }

    /* กล่องผลลัพธ์ */
    .stAlert {
        background-color: #352F33 !important;
        color: #FFFFFF !important;
        border-radius: 10px !important;
        border: 1px solid #D72638 !important;
    }

    /* caption ด้านล่าง */
    .stCaption {
        color: #BBBBBB;
    }
    </style>
""", unsafe_allow_html=True)

st.caption("โปรเจกต์นี้จัดทำเพื่อการศึกษาในรายวิชา Industrial Artificial Intelligence and IoT (01211271) เท่านั้น")