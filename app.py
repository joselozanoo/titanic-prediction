import streamlit as st
import joblib
import pandas as pd

# Intenta cargar el modelo
try:
    model = joblib.load("./modelo.pkl")
    st.success("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    model = None  # Evita que la app falle si no se carga el modelo
    
# Título
st.title("Predicción de Supervivencia en el Titanic")

# Campos de entrada
pclass = st.selectbox("Clase del pasajero", [1, 2, 3])
sex = st.selectbox("Sexo", ["male", "female"])
age = st.number_input("Edad", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Número de hermanos/esposo(a) a bordo", min_value=0, max_value=10, value=0)
parch = st.number_input("Número de padres/hijos a bordo", min_value=0, max_value=10, value=0)
fare = st.number_input("Tarifa pagada", min_value=0.0, max_value=500.0, value=50.0)
embarked = st.selectbox("Puerto de embarque", ["C", "Q", "S"])

# Mapeo valores categóricos a numéricos
sex = 0 if sex == 'female' else 1
embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
embarked = embarked_dict[embarked]

# Botón para realizar la predicción
if st.button("Predecir supervivencia"):
    # Creo un DataFrame con los datos ingresados
    input_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                              columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    
    # Realizo la predicción
    prediction = model.predict(input_data)
    
    # Muestro el resultado
    resultado = "El pasajero ha sobrevivido." if prediction[0] == 1 else "El pasajero no ha sobrevivido."
    st.success(resultado)
