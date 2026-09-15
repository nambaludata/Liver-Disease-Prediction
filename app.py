

#start of code

import streamlit as st # for creating the web app using python
import joblib # allows us to load the trained model and preprocessing
import pandas as pd



st.set_page_config(
    page_title="Liver Disease Prediction App",
    page_icon="🩺",
    layout="centered"
)




final_model = joblib.load('rm.pkl') # now our application is able to access the trained model
feature_names = joblib.load('feature_names.pkl') # we also get to load the feature names so the app can access




st.title("🩺 Liver Disease Prediction App") # Create the app title


st.write(
    "Enter the patient's clinical information below "
    "to generate a machine-learning prediction."
)


 # Create a description that will be deployed

st.info(
    "This application is for educational and demonstration purposes. "
    "The prediction should not be considered a medical diagnosis."
)

st.divider()


st.header("👤 Patient Information")

# We then go ahead and create Input Values for every feature
col1, col2 = st.columns(2)

with col1:
  age = st.number_input( "Age", min_value=1, max_value=120, value=30)

with col2:
  gender = st.selectbox("Gender", ["Female", "Male"])

gender_male = 1 if gender == "Male" else 0


st.divider()

st.header("🧪 Laboratory Results")
col1, col2 = st.columns(2)

with col1:
  total_bilirubin = st.number_input("Total_Bilirubin", min_value=0.0)

with col2:
  direct_bilirubin = st.number_input("Direct_Bilirubin",min_value=0.0)

with col1:
  alkaline_phosphotase = st.number_input("Alkaline_Phosphotase", min_value=0.0)

with col2:
  alamine_aminotransferase = st.number_input("Alamine_Aminotransferase",min_value=0.0)

with col1:
  aspartate_aminotransferase = st.number_input("Aspartate_Aminotransferase",min_value=0.0)

with col2:
  total_proteins = st.number_input("Total_Protiens",min_value=0.0)

with col1:
  albumin = st.number_input("Albumin",min_value=0.0)

with col2:
  albumin_globulin_ratio = st.number_input("Albumin_and_Globulin_Ratio", min_value=0.0)





#make a list for all the input data features

input_data = pd.DataFrame({
        "Age": [age],
        "Total_Bilirubin": [total_bilirubin],
        "Direct_Bilirubin": [direct_bilirubin],
        "Alkaline_Phosphotase": [alkaline_phosphotase],
        "Alamine_Aminotransferase": [alamine_aminotransferase],
        "Aspartate_Aminotransferase": [aspartate_aminotransferase],
        "Total_Protiens": [total_proteins],
        "Albumin": [albumin],
        "Albumin_and_Globulin_Ratio": [albumin_globulin_ratio],
        "Gender_Male": [gender_male]
 })

input_data = input_data[feature_names] # match the order for the features



# Creating a button on the app

st.divider() 

if st.button("🔮 Predict Liver Disease", use_container_width=True):
     prediction = final_model.predict(input_data) # prediction
     probability = final_model.predict_proba(input_data)
     disease_probability = probability[0][1] * 100


     st.subheader("🩺 Prediction Result")


     if prediction[0] == 1:
         st.error("⚠️ Higher likelihood of liver disease")

         st.write(
            "Based on the information provided, the model predicts "
            "a higher likelihood of liver disease."
         )

         st.metric("Model Probability",f"{disease_probability:.1f}%")


     else:
         st.success("✅ Lower likelihood of liver disease")

         st.write(
            "Based on the information provided, the model predicts "
            "a lower likelihood of liver disease."
         )

         st.metric("Model Probability",f"{disease_probability:.1f}%")


# end of code

