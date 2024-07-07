import streamlit as st
import pandas as pd
import joblib  # Use joblib to load the model

# Load your trained model
joblib_file = "model_testing.pkl"
loaded_model = joblib.load(joblib_file)

# Define the min and max values for the numerical columns
numerical_columns_min_max = {
    'Age': (18, 79),
    'NumOpioidPrescriptions': (1, 19),
    'AverageDosage': (5, 99),
    'DurationOfPrescriptions': (1, 29),
    'NumHealthcareVisits': (0, 19),
    'NumHospitalizations': (0, 4),
    'Duration': (1, 29),
    'Refills': (0, 4),
    'TotalTimeSpentwithPhysician': (10, 59)
}

# Define the unique classes for the categorical columns
categorical_columns_classes = {
    'Gender': ['Female', 'Male'],
    'Race': ['Other', 'Asian', 'White', 'Black', 'Hispanic'],
    'ChronicPainConditions': ['Fibromyalgia', 'Cancer Pain', 'Post-Surgery Pain', 'Chronic Back Pain', 'Arthritis'],
    'PainManagementTreatment': ['No', 'Yes'],
    'MedicationName': ['Hydrocodone', 'Hydromorphone', 'Oxymorphone', 'Tramadol', 'Codeine', 'Morphine', 'Buprenorphine', 'Oxycodone', 'Meperidine', 'Methadone', 'Fentanyl', 'Tapentadol'],
    'Dosage': ['10 mg', '100 mcg/hour', '20 mg', '80 mg', '60 mg', '12.5 mcg/hour', '50 mcg/hour', '40 mg', '2.5 mg', '75 mcg/hour', '5 mg', '30 mg'],
    'Frequency': ['every 8 hours', 'every 4-6 hours', 'every 12 hours', 'once daily'],
    'MedicationClass': ['Opioid', 'Narcotic', 'Analgesic'],
    'Adherence': ['Moderate', 'Low', 'High'],
    'ClinicalNotes': ['Post-operative pain managed with Hydrocodone.', 'Prescribed Oxymorphone for severe pain.', 'Patient reports effective pain relief with Tapentadol.', 'Using Tramadol for moderate pain management.', 'Patient with previous substance abuse, now on Methadone treatment.', 'Recovering from injury, prescribed Codeine.', 'Experiencing severe pain from cancer, prescribed Morphine.', 'History of depression, taking Oxycodone for post-surgery pain.', 'Patient reports chronic back pain, prescribed Fentanyl patch.', 'Meperidine prescribed for acute pain episodes.', 'Chronic arthritis, using Hydromorphone for pain relief.'],
    'Specialty': ['Orthopedics', 'Pain Management', 'Oncology', 'Primary Care'],
    'AppointmentType': ['Routine Check-up', 'Consultation', 'Follow-up'],
    'SubSpecialty': ['Specialized', 'General']
}

# Title of the Streamlit app
st.title('Predicting Patients at High Risk of Opioid Crisis by SynapseHealthTech ')

# User input for numerical columns
st.header('Numerical Input')
numerical_input = {}
for column, (min_val, max_val) in numerical_columns_min_max.items():
    numerical_input[column] = st.slider(f'{column}', min_val, max_val, (min_val + max_val) // 2)

# User input for categorical columns
st.header('Categorical Input')
categorical_input = {}
for column, classes in categorical_columns_classes.items():
    categorical_input[column] = st.selectbox(f'{column}', classes)

# Display the input values
st.subheader('Your Input Values')
st.write('Numerical Input:', numerical_input)
st.write('Categorical Input:', categorical_input)

# Preprocess the input DataFrame
# Define mappings for categorical columns
categorical_mappings = {
    'Gender': {'Female': 0, 'Male': 1},
    'Race': {'Other': 0, 'Asian': 1, 'White': 2, 'Black': 3, 'Hispanic': 4},
    'ChronicPainConditions': {'Fibromyalgia': 0, 'Cancer Pain': 1, 'Post-Surgery Pain': 2, 'Chronic Back Pain': 3, 'Arthritis': 4},
    'PainManagementTreatment': {'No': 0, 'Yes': 1},
    'MedicationName': {'Hydrocodone': 0, 'Hydromorphone': 1, 'Oxymorphone': 2, 'Tramadol': 3, 'Codeine': 4, 'Morphine': 5, 'Buprenorphine': 6, 'Oxycodone': 7, 'Meperidine': 8, 'Methadone': 9, 'Fentanyl': 10, 'Tapentadol': 11},
    'Dosage': {'10 mg': 0, '100 mcg/hour': 1, '20 mg': 2, '80 mg': 3, '60 mg': 4, '12.5 mcg/hour': 5, '50 mcg/hour': 6, '40 mg': 7, '2.5 mg': 8, '75 mcg/hour': 9, '5 mg': 10, '30 mg': 11},
    'Frequency': {'every 8 hours': 0, 'every 4-6 hours': 1, 'every 12 hours': 2, 'once daily': 3},
    'MedicationClass': {'Opioid': 0, 'Narcotic': 1, 'Analgesic': 2},
    'Adherence': {'Moderate': 0, 'Low': 1, 'High': 2},
    'ClinicalNotes': {'Post-operative pain managed with Hydrocodone.': 0, 'Prescribed Oxymorphone for severe pain.': 1, 'Patient reports effective pain relief with Tapentadol.': 2, 'Using Tramadol for moderate pain management.': 3, 'Patient with previous substance abuse, now on Methadone treatment.': 4, 'Recovering from injury, prescribed Codeine.': 5, 'Experiencing severe pain from cancer, prescribed Morphine.': 6, 'History of depression, taking Oxycodone for post-surgery pain.': 7, 'Patient reports chronic back pain, prescribed Fentanyl patch.': 8, 'Meperidine prescribed for acute pain episodes.': 9, 'Chronic arthritis, using Hydromorphone for pain relief.': 10},
    'Specialty': {'Orthopedics': 0, 'Pain Management': 1, 'Oncology': 2, 'Primary Care': 3},
    'AppointmentType': {'Routine Check-up': 0, 'Consultation': 1, 'Follow-up': 2},
    'SubSpecialty': {'Specialized': 0, 'General': 1}
}

# Function to preprocess the input DataFrame
def preprocess_input(df):
    # Create a new DataFrame for preprocessing
    processed_df = df.copy()

    # Map categorical columns to numerical values
    for column, mapping in categorical_mappings.items():
        processed_df[column] = processed_df[column].map(mapping).fillna(len(mapping))  # Use len(mapping) for unknown values

    return processed_df

# Create a DataFrame from the input values in the correct column order
input_data = {**numerical_input, **categorical_input}
columns_order = [
    'Age', 'Gender', 'Race', 'ChronicPainConditions', 'NumOpioidPrescriptions', 'AverageDosage', 
    'DurationOfPrescriptions', 'NumHealthcareVisits', 'NumHospitalizations', 'PainManagementTreatment', 
    'MedicationName', 'Dosage', 'Frequency', 'Duration', 'Refills', 'MedicationClass', 'Adherence', 
    'ClinicalNotes', 'Specialty', 'AppointmentType', 'SubSpecialty', 'TotalTimeSpentwithPhysician'
]
input_df = pd.DataFrame([input_data], columns=columns_order)

# Preprocess the input DataFrame
processed_input_df = preprocess_input(input_df)


# Display the processed input values
# st.subheader('Processed Input Values')
# st.write(processed_input_df)

# Make prediction

if st.button('Predict'):
  new_prediction = loaded_model.predict(processed_input_df)
    # Display prediction result
  st.subheader('Prediction Result:')
  if new_prediction[0] == 0:
      st.title("Low risk")
  elif new_prediction[0] == 1:
      st.title("High risk")