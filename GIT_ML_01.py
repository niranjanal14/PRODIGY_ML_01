import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Correctly formatted file path to the dataset using raw string notation
file_path = r'C:\Users\niran\OneDrive\Dokumen\home.csv'

# Check if the file exists
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"The file at path {file_path} does not exist. Please check the file path.")

# Load the dataset with the correct encoding (if needed)
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='latin1')  # Try an alternative encoding

# Features and target variable
X = df[['Square Footage', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Ensure X_test is a DataFrame with the correct feature names
X_test = pd.DataFrame(X_test, columns=['Square Footage', 'Bedrooms', 'Bathrooms'])

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Print coefficients
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Ask the user for input values
square_footage = float(input("Enter the square footage: "))
bedrooms = int(input("Enter the number of bedrooms: "))
bathrooms = int(input("Enter the number of bathrooms: "))

# Create a DataFrame for the input values
input_data = pd.DataFrame([[square_footage, bedrooms, bathrooms]], columns=['Square Footage', 'Bedrooms', 'Bathrooms'])

# Make a prediction using the input values
predicted_price = model.predict(input_data)

print(f'Predicted Price: {predicted_price[0]}')
