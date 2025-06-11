import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Generate Synthetic Dataset

np.random.seed(42)  # For reproducibility

# Generate 500 samples
n_samples = 500

# Square footage between 500 and 3500 sqft
sqft = np.random.randint(500, 3500, size=n_samples)

# Bedrooms: 1 to 5
bedrooms = np.random.randint(1, 6, size=n_samples)

# Bathrooms: 1 to 4
bathrooms = np.random.randint(1, 5, size=n_samples)

price = (
    50000 +
    (150 * sqft) +
    (10000 * bedrooms) +
    (15000 * bathrooms) +
    np.random.normal(0, 20000, n_samples)  # noise
)

# Create DataFrame
df = pd.DataFrame({
    'sqft': sqft,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'price': price
})

# Step 2: Exploratory Data Analysis (EDA)

print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset statistics:")
print(df.describe())

# Plot distribution of target variable
plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# Plot relationships between features and price
sns.pairplot(df)
plt.suptitle("Pairplot of Features and Price", y=1.02)
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Step 3: Prepare Data for Modeling

X = df[['sqft', 'bedrooms', 'bathrooms']]
y = df['price']

# Split dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Step 4: Build and Train Linear Regression Model

model = LinearRegression()

# Fit model on training data
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Step 5: Model Evaluation on Test Set

y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance on Test Set:")
print(f"Mean Squared Error (MSE): {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"Mean Absolute Error (MAE): {mae:,.2f}")
print(f"R² Score: {r2:.4f}")

# Print model coefficients
print("\nModel Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Step 6: Visualize Predictions

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Residual (Actual - Predicted)")
plt.show()

plt.figure(figsize=(8,6))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Prices")
plt.show()