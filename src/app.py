# your code here
import streamlit as st
import joblib

model = joblib.load("/workspaces/ML-web-app-using-Streamlit-gregoriant/src/decision_tree_classifier_default_42.pkl")
class_dict = {"0": "No Diabético",
              "1": "Diabético"}
st.title("Diabetes - Model prediciton")    
        
# Obtain values from form
val1 = st.slider('Glucose', min_value = 50.0, max_value = 200.0, step = 1.0)
val2 = st.slider('BloodPressure', min_value = 40.0, max_value = 120.0, step = 1.0)
val3 = st.slider('SkinThickness', min_value = 10.0, max_value = 60.0, step = 1.0)
val4 = st.slider('Insulin', min_value = 2.0, max_value = 250.0, step = 1.0)
val5 = st.slider('BMI', min_value = 15.0, max_value = 50.0, step = 1.0)
val6 = st.slider('Age', min_value = 18.0, max_value = 90.0, step = 1.0)

if st.button("Predict"):
    data = [[val1, val2, val3, val4, val5, val6]]
    prediction = str(model.predict(data)[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)