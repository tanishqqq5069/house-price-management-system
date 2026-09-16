import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

print("Step 1: Generating Massive Greater Noida Dataset...")
# A comprehensive list of Greater Noida sectors
locations = [
    "Alpha 1", "Alpha 2", "Beta 1", "Beta 2", "Gamma 1", "Gamma 2", 
    "Delta 1", "Omicron 1", "Zeta 1", "Eta 1", "Knowledge Park", 
    "Techzone 4", "Sector 1", "Sector 2", "Sector 4", "Sector 10", 
    "Sector 12", "Sector 16", "Sector 16B", "Sector 150"
]

# Generate 500 rows of random but realistic data
np.random.seed(42)
data = {
    'Area_SqFt': np.random.randint(800, 3000, 500),
    'BHK': np.random.randint(1, 5, 500),
    'Age_Years': np.random.randint(0, 20, 500),
    'Location': np.random.choice(locations, 500),
}
df = pd.DataFrame(data)

# Create a realistic pricing formula based on features
df['Price_Lakhs'] = (df['Area_SqFt'] * 0.05) + (df['BHK'] * 5) - (df['Age_Years'] * 0.5) + np.random.randint(5, 20, 500)
print(f"Generated {len(df)} records across {len(locations)} sectors.\n")

print("Step 2: Preprocessing and Encoding...")
# Convert all locations into 0s and 1s automatically
df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)

X = df_encoded.drop('Price_Lakhs', axis=1)
y = df_encoded['Price_Lakhs']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Step 3: Training Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Step 4: Saving Model and Column Structure...")
# CRITICAL: We save the model AND the exact column names it learned, 
# along with the full list of locations for the dropdown menu!
model_data = {
    'model': model,
    'expected_columns': X.columns,
    'all_locations': locations
}

with open('house_price_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)
    
print("Success! Automated model is ready.")