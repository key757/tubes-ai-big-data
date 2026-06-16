import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import StandardScaler # For type hinting and understanding

# --- Configuration --- 
st.set_page_config(
    page_title="Prediksi Risiko Stroke",
    page_icon="🧠",
    layout="centered"
)

# --- Load Model and Preprocessors ---
try:
    with open('model_terbaik.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('info_model.json', 'r') as f:
        info_model = json.load(f)
    # Pastikan scaler.pkl juga disimpan. Jika belum, jalankan kode berikut di notebook Anda:
    # import pickle
    # from sklearn.preprocessing import StandardScaler
    # # Pastikan objek `scaler` sudah ada dan sudah fit dari proses pelatihan model
    # with open('scaler.pkl', 'wb') as f:
    #     pickle.dump(scaler, f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

except FileNotFoundError:
    st.error("Pastikan file 'model_terbaik.pkl', 'info_model.json', dan 'scaler.pkl' ada di direktori yang sama.")
    st.stop()

# Get feature names from info_model (to ensure correct order for prediction)
features_order = info_model['fitur']

# --- Hardcoded Categorical Mappings (based on LabelEncoder alphabetical order) ---
# Ini diperlukan karena objek LabelEncoder asli digunakan kembali dan tidak disimpan secara individual.
# Urutan mapping didasarkan pada urutan abjad yang digunakan oleh LabelEncoder.
gender_map = {'Female': 0, 'Male': 1, 'Other': 2}
ever_married_map = {'No': 0, 'Yes': 1}
work_type_map = {'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4}
residence_type_map = {'Rural': 0, 'Urban': 1}
smoking_status_map = {'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3}

# --- Feature Engineering Functions (replicated from notebook) ---
def get_age_group(age):
    if age < 18:
        return 'Anak'
    elif age < 40:
        return 'Dewasa Muda'
    elif age < 60:
        return 'Dewasa'
    else:
        return 'Lansia'

def get_bmi_category(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obesitas'

# Mapping for age_group and bmi_category (after they are generated as strings)
age_group_map = {'Anak': 0, 'Dewasa': 1, 'Dewasa Muda': 2, 'Lansia': 3}
bmi_category_map = {'Normal': 0, 'Obesitas': 1, 'Overweight': 2, 'Underweight': 3}


# --- Streamlit UI ---
st.title("🧠 Prediksi Risiko Stroke")
st.write("Aplikasi ini memprediksi risiko stroke berdasarkan data kesehatan pasien.")

st.sidebar.header("Input Data Pasien")

# Input fields for features
gender = st.sidebar.selectbox("Jenis Kelamin", list(gender_map.keys()))
age = st.sidebar.slider("Usia", 0.08, 82.0, 40.0) # From df.describe() min/max age
hypertension = st.sidebar.selectbox("Hipertensi", ['Tidak', 'Ya'], format_func=lambda x: 'Ya' if x == 'Ya' else 'Tidak')
heart_disease = st.sidebar.selectbox("Penyakit Jantung", ['Tidak', 'Ya'], format_func=lambda x: 'Ya' if x == 'Ya' else 'Tidak')
ever_married = st.sidebar.selectbox("Sudah Menikah", list(ever_married_map.keys()))
work_type = st.sidebar.selectbox("Tipe Pekerjaan", list(work_type_map.keys()))
residence_type = st.sidebar.selectbox("Tipe Tempat Tinggal", list(residence_type_map.keys()))
avg_glucose_level = st.sidebar.slider("Rata-rata Kadar Glukosa", 21.98, 169.36, 100.0) # Capped values from notebook
bmi = st.sidebar.slider("Indeks Massa Tubuh (BMI)", 10.3, 46.3, 25.0) # Capped values from notebook
smoking_status = st.sidebar.selectbox("Status Merokok", list(smoking_status_map.keys()))


# --- Prediction Button ---
if st.sidebar.button("Prediksi Risiko"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame([[
        gender, age, hypertension, heart_disease, ever_married, work_type,
        residence_type, avg_glucose_level, bmi, smoking_status
    ]], columns=[
        'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
        'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
    ])

    # Apply Feature Engineering
    input_data['age_group'] = input_data['age'].apply(get_age_group)
    input_data['bmi_category'] = input_data['bmi'].apply(get_bmi_category)

    # Apply Categorical Mappings (manual LabelEncoding)
    input_data['gender'] = input_data['gender'].map(gender_map)
    input_data['ever_married'] = input_data['ever_married'].map(ever_married_map)
    input_data['work_type'] = input_data['work_type'].map(work_type_map)
    input_data['Residence_type'] = input_data['Residence_type'].map(residence_type_map)
    input_data['smoking_status'] = input_data['smoking_status'].map(smoking_status_map)
    input_data['age_group'] = input_data['age_group'].map(age_group_map)
    input_data['bmi_category'] = input_data['bmi_category'].map(bmi_category_map)

    # Convert 'Ya'/'Tidak' for hypertension and heart_disease to 1/0
    input_data['hypertension'] = input_data['hypertension'].apply(lambda x: 1 if x == 'Ya' else 0)
    input_data['heart_disease'] = input_data['heart_disease'].apply(lambda x: 1 if x == 'Ya' else 0)

    # Select numerical columns for scaling
    numerical_cols_for_scaling = ['age', 'avg_glucose_level', 'bmi']
    input_data[numerical_cols_for_scaling] = scaler.transform(input_data[numerical_cols_for_scaling])

    # Ensure the order of features is correct for the model
    final_input = input_data[features_order]

    # Make prediction
    prediction = model.predict(final_input)
    prediction_proba = model.predict_proba(final_input)[:, 1] # Probability of stroke (class 1)

    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.error(f"**Risiko Stroke TINGGI!** (Probabilitas: {prediction_proba[0]*100:.2f}%)")
        st.write("Sangat disarankan untuk segera berkonsultasi dengan dokter.")
    else:
        st.success(f"**Risiko Stroke RENDAH.** (Probabilitas: {prediction_proba[0]*100:.2f}%)")
        st.write("Meskipun risiko rendah, tetap jaga gaya hidup sehat.")

    st.write("---")

st.subheader("Informasi Model Terbaik")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model", info_model['nama_model'])
with col2:
    st.metric("Accuracy", f"{info_model['accuracy']:.4f}")
with col3:
    st.metric("Precision", f"{info_model['precision']:.4f}")
with col4:
    st.metric("Recall", f"{info_model['recall']:.4f}")
st.metric("F1-Score", f"{info_model['f1_score']:.4f}")
st.write(f"Fitur yang digunakan: {', '.join(info_model['fitur'])}")

st.caption("Pastikan `model_terbaik.pkl`, `info_model.json`, dan `scaler.pkl` berada di direktori yang sama dengan aplikasi Streamlit ini.")
st.caption("Jika `scaler.pkl` belum disimpan, jalankan kode berikut di notebook Anda (setelah menjalankan sel-sel preprocessing):")
st.code("""
import pickle
from sklearn.preprocessing import StandardScaler

# Jika objek `scaler` masih ada dalam memori:
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Jika Anda perlu membuat ulang dan melatih `scaler`:
# scaler = StandardScaler()
# numerical_cols = ['age', 'avg_glucose_level', 'bmi'] # Gunakan kolom yang sama seperti saat pelatihan
# scaler.fit(df[numerical_cols]) # Pastikan df adalah dataframe yang sudah diproses sebelum encoding
# with open('scaler.pkl', 'wb') as f:
#     pickle.dump(scaler, f)
""", language="python")
