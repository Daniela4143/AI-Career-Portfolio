from sklearn.datasets import make_regression # For generating a regression dataset
import pandas as pd # For data manipulation
from sklearn.preprocessing import StandardScaler # For feature scaling
from sklearn.linear_model import LinearRegression # For linear regression model
from sklearn.metrics import mean_squared_error, r2_score # For model evaluation
import numpy as np # For numerical operations
from sklearn.model_selection import train_test_split # For splitting dataset
import matplotlib.pyplot as plt # For plotting
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

def load_and_prepare_data():
    '''Load and prepare the dataset for regression analysis.'''

    print("1.Loading and preparing data...")

    # Generate a synthetic regression dataset
    # X: feature matrix, y: target variable
    X, y = make_regression(n_samples=1000, n_features=10, n_informative=5,
                           noise=10, random_state=42)
    
    # Convert to DataFrame for easier handling
    feature_names = [f'Feature_{i+1}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['Target_Price'] = y # Add target variable to DataFrame

    print(f"Dataset Shape: {df.shape}")
    print(df.head())

    # Separate features and target variable
    X = df.drop('Target_Price', axis=1)
    y = df['Target_Price']

    return X, y

def preprocess_data(X_train, X_test):
    '''
    Use StandardScaler to standardize features, 
    which is a important preprocessing step for ML models.
    '''

    print("2.Preprocessing data (scaling features)...")

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Only (fit) Scaler on training dataset to avoid data leakage
    X_train_scaled = scaler.fit_transform(X_train)

    # Use Scaler that was fit on training data to transform test data
    X_test_scaled = scaler.transform(X_test)

    print("Feature standardization complete.")
    
    return X_train_scaled, X_test_scaled

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    '''
    Train a linear regression model and evaluate its performance on test dataset.
    '''

    print("3.Training and evaluating the model...")

    # Initialize the Linear Regression model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)
    print("Linear Regression model training complete.")

    # Predict on the test dataset
    y_pred = model.predict(X_test)

    # Evaluate model performance
    # MSE (Mean Squared Error)
    mse = mean_squared_error(y_test, y_pred)
    # RMSE (Root Mean Squared Error) - the indicator more interpretable whose unit is same as target variable
    rmse = np.sqrt(mse)
    # R-squared (Coefficient of Determination) - measure the capability of the model to explain variance (the better the model, the closer to 1)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel Evaluation Results (Linear Regression):")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  Coefficient of Determination (R-squared): {r2:.2f}")

    return y_test, y_pred

def plot_results(y_test, y_pred):
    '''
    Visualize the comparison plot between actual and predicted target values.
    '''
    
    print("\n4.Visualizing prediction results...")

    # scatter plot: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, color='darkblue')

    # Plotting the ideal prediction line (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val],
             '--r', label='Perfect Prediction')
    
    plt.title('Actual vs Predicted Price', fontsize=14)
    plt.xlabel('Actual Price (Target Price)', fontsize=12)
    plt.ylabel('Predicted Price', fontsize=12)
    plt.legend()
    plt.show()

    # 新增一個殘差分佈圖
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residuals Distribution (Error Analysis)', fontsize=14)
    plt.xlabel('Prediction Error', fontsize=12)
    plt.show()

def main():
    # step 1: Load and prepare data
    X, y = load_and_prepare_data()

    # step 2: Split dataset (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training Set Size: {X_train.shape}, Test Set Size: {X_test.shape}")

    # step 3: Preprocess data (feature scaling)
    X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test)

    # step 4: Train and evaluate model
    y_test_actual, y_test_predicted = train_and_evaluate_model(
        X_train_scaled, y_train, X_test_scaled, y_test
    )

    # Step 5: Visualize results
    plot_results(y_test_actual, y_test_predicted)

if __name__ == "__main__":
    main()



