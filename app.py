import streamlit as st
import pandas as pd
import pickle

# 1. Load the model, the expected columns, and the location list
with open('house_price_model.pkl', 'rb') as file:
    model_data = pickle.load(file)
    
model = model_data['model']
expected_columns = model_data['expected_columns']
locations = sorted(model_data['all_locations']) # Alphabetize the dropdown

# 2. Design the Web Page
st.title("🏡 Greater Noida Real Estate Predictor")
st.write("Enter the property details below to get an estimated market price.")

# 3. Create Input Fields
area = st.number_input("Area (in SqFt)", min_value=500, max_value=5000, value=1200)
bhk = st.selectbox("Number of Bedrooms (BHK)", [1, 2, 3, 4, 5])
age = st.slider("Property Age (in Years)", 0, 50, 5)

# The dropdown now populates automatically from the list we saved!
location = st.selectbox("Select Greater Noida Sector", locations)

# 4. Make Predictions
if st.button("Predict Price"):
    
    # Create an empty DataFrame exactly how the model expects it, filled with 0s
    input_df = pd.DataFrame(columns=expected_columns)
    input_df.loc[0] = 0 
    
    # Insert the user's basic inputs
    input_df['Area_SqFt'] = area
    input_df['BHK'] = bhk
    input_df['Age_Years'] = age
    
    # Flip the 0 to a 1 only for the location the user selected
    loc_column_name = f'Location_{location}'
    if loc_column_name in input_df.columns:
        input_df[loc_column_name] = 1
        
    # Generate the prediction
    predicted_price = model.predict(input_df)[0]
    st.success(f"Estimated Price: ₹ {predicted_price:.2f} Lakhs")