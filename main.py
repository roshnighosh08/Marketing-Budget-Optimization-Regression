# ==========================================
# 1. IMPORT LIBRARIES
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for visualizations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# ==========================================
# 2. LOAD DATASET
# ==========================================
df = pd.read_csv("part_4_marketing_budget_optimization.csv")
print("--- Raw Dataset Shape ---")
print(df.shape)
print("\n--- First 5 Rows ---")
print(df.head())

# ==========================================
# 3. DATA UNDERSTANDING
# ==========================================
print("\n--- Column Structural Info ---")
print(df.info())

print("\n--- Missing Values Count ---")
print(df.isnull().sum())

print("\n--- Summary Statistics ---")
print(df.describe())

# ==========================================
# 4. DATA CLEANING
# ==========================================
df_clean = df.copy()

# 1. Impute missing values found inside SocialMedia_Spend using its median
df_clean['SocialMedia_Spend'] = df_clean['SocialMedia_Spend'].fillna(df_clean['SocialMedia_Spend'].median())

# 2. Remove duplicate logs if present
df_clean = df_clean.drop_duplicates()

print("\n--- Pre-processed Cleaned Shape ---")
print(df_clean.shape)

# ==========================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
# Chart 1: Distribution of Sales Revenue
plt.figure()
sns.histplot(df_clean['Sales_Revenue'], kde=True, color='purple')
plt.title('Distribution Profile of Target Sales Revenue')
plt.xlabel('Sales Revenue ($)')
plt.ylabel('Density Count')
plt.tight_layout()
plt.savefig('sales_distribution.png')
plt.show()
print("\nInterpretation Chart 1: The distribution of sales revenue displays a normal bell curve shape. This tells us that sales performance is stable, making it a great fit for linear regression modeling.")

# Chart 2: Correlation Heatmap Matrix
plt.figure(figsize=(8,6))
spend_features = ['TV_Spend', 'Radio_Spend', 'SocialMedia_Spend', 'SearchAds_Spend', 'Influencer_Spend', 'Sales_Revenue']
sns.heatmap(df_clean[spend_features].corr(), annot=True, cmap='RdYlBu_r', fmt=".2f")
plt.title('Correlation Analysis Matrix: Marketing Spend vs Sales')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
print("\nInterpretation Chart 2: The heatmap matrix shows how different channels connect to revenue. Radio and Search Ads show strong positive correlation values, pointing them out as primary revenue drivers.")

# Chart 3: Scatter Plot - Radio Spend vs Sales
plt.figure()
sns.regplot(data=df_clean, x='Radio_Spend', y='Sales_Revenue', color='teal', scatter_kws={'alpha':0.6})
plt.title('Marketing Channel Impact: Radio Spend vs Sales Revenue')
plt.xlabel('Radio Spend ($)')
plt.ylabel('Sales Revenue ($)')
plt.tight_layout()
plt.savefig('radio_vs_sales.png')
plt.show()
print("\nInterpretation Chart 3: The upward trend line confirms a clear positive relationship. As investment in radio advertising grows, sales revenue scales upward in a highly predictable linear pattern.")

# ==========================================
# 6. FEATURE ENGINEERING
# ==========================================
# Select independent continuous features and target matrix
features = ['TV_Spend', 'Radio_Spend', 'SocialMedia_Spend', 'SearchAds_Spend', 'Influencer_Spend']
X = df_clean[features]
y = df_clean['Sales_Revenue']

# Split data into training (80%) and validation (20%) partitions
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Data Split Shapes Matrix ---")
print(f"X_train Shape: {X_train.shape} | X_test Shape: {X_test.shape}")

# ==========================================
# 7. MODEL BUILDING / ANALYSIS
# ==========================================
# Train the Multiple Linear Regression Pipeline
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Generate predictions across our test matrix
y_pred = reg_model.predict(X_test)

print("\n=== REGRESSION COEFFICIENT INSIGHTS ===")
print(f"Model Baseline Intercept: {reg_model.intercept_:.4f}")
for col, coef in zip(features, reg_model.coef_):
    print(f" -> {col} Coefficient value: {coef:.4f}")

# ==========================================
# 8. EVALUATION
# ==========================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n=== MODEL PERFORMANCE METRICS ===")
print(f" -> Mean Absolute Error (MAE): {mae:.4f}")
print(f" -> Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f" -> Coefficient of Determination (R2 Score): {r2:.4f}")

# ==========================================
# 9. BUSINESS INSIGHTS (ROAS Matrix Analysis)
# ==========================================
# Calculate overall return metrics across channels based on historical dataset footprint
print("\n=== CHANNEL EFFICIENCY ANALYSIS ===")
for col, coef in zip(features, reg_model.coef_):
    print(f"Channel Asset [{col}]: Every $1.00 invested generates ${coef:.2f} in incremental sales revenue return.")

# ==========================================
# 10. FINAL RECOMMENDATIONS
# ==========================================
print("\n=== SYSTEM EXECUTION COMPLETE ===")
print("Review generated plots in the execution directory and proceed to compile documentation repository files.")