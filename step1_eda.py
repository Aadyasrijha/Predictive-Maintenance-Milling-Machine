import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the raw data
# (Assuming the file is named 'ai4i2020.csv' and is in the same folder)
df = pd.read_csv('ai4i2020.csv')
# 2. The Integrity Check (Looking for missing or broken data)
print("--- DATA INTEGRITY CHECK ---")
# This tells us if any sensor went offline and recorded a "null" value
print(df.isnull().sum()) 
# 3. The Statistical Profile (Looking for physical impossibilities)
print("\n--- STATISTICAL PROFILE ---")
# This calculates the Min, Max, and Average for every sensor. 
# If the 'Min' RPM is negative, we know we have a broken sensor.
print(df.describe())
# 4. The Physics Check: Correlation Matrix
# We want to see how the sensors interact with each other mathematically.
# A score of 1.0 means they move perfectly together. 
# A score of -1.0 means they move exactly opposite.
plt.figure(figsize=(10, 8))
# We drop non-numeric columns like Product ID before correlating
numeric_df = df.drop(['UDI', 'Product ID', 'Type'], axis=1)

# Generate the heatmap
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Sensor Correlation Heatmap")
plt.show()