import pandas as pd
import numpy as np

# 1. Load the dataset from your local folder
df = pd.read_csv('ai4i2020.csv')

print("--- RAW DATA SHAPE ---")
print(df.shape) # Shows how many rows and columns we start with

# 2. Feature Engineering: Adding Mechanical Physics

# A. Temperature Difference (Heat Dissipation Check)
df['Temp_Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

# B. Mechanical Power (Power = Torque * Angular Velocity)
# Converting RPM to radians per second: RPM * (2 * pi / 60)
df['Power_W'] = df['Torque [Nm]'] * (df['Rotational speed [rpm]'] * (2 * np.pi / 60))

# C. Overstrain Metric (Tool wear multiplied by torque)
df['Strain_Metric'] = df['Tool wear [min]'] * df['Torque [Nm]']

print("\n--- ENGINEERED DATA SAMPLE ---")
# Display the newly created engineering metrics side-by-side
print(df[['Temp_Difference', 'Power_W', 'Strain_Metric']].head())

# 3. Save this enhanced data to a new file for our model training step
df.to_csv('ai4i2020_engineered.csv', index=False)
print("\nSuccess! Engineered dataset saved as 'ai4i2020_engineered.csv'.")