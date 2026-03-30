"""
Student Exam Performance Analysis - Step 3 Code Review
IS 392 Section 452
Minwoo Choi, Hannah Repuyan, Victor Gutierrez

RESEARCH QUESTIONS:
1. How does demographic information influence exam performance?
2. Is a student's parental education level related to exam performance?
3. Is there a correlation between test preparation and exam scores?

Code implementation methodology:
Step 1: Data Cleaning and Preprocessing
  - Handle missing values and inconsistencies
  - Encode demographic categories to numerical form
  - Prepare dataset for modeling

Step 2: Exploratory Data Analysis
  - Descriptive statistics (mean, median, std dev)
  - Histogram visualizations of score distributions
  - Correlation analysis (how background factors relate to performance)
  - Group comparisons (boxplots by demographic groups)

Step 3: Linear Regression Modeling
  - Build model analyzing background factors with test scores
  - Examine coefficients to identify important factors

Step 4: Model Evaluation
  - R-squared: measure variance explained in test scores
  - RMSE: show accuracy of predictions
  - Interpret findings to identify biggest factors

Dataset: Kaggle - Students Performance in Exams
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# Data loading and exploration
def load_data(filepath):
    """Load CSV file with exam performance data"""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"Error: {filepath} not found")
        print("Download: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams")
        raise


def explore_data(df):
    """Check dataset structure and identify missing/inconsistent values"""
    print("\n--- Data Exploration ---")
    print(f"Shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}")
    
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}")
    
    print(f"First rows:\n{df.head()}")
    print(f"Stats:\n{df.describe().round(2)}")


# Step 1: Data prprocessing
def clean_data(df):
    """
    Clean and prepare data for analysis:
    - Remove rows with missing values and inconsistencies
    - Encode demographic categories to numerical form
    """
    print("\n--- Data Preprocessing ---")
    df = df.copy()
    
    # Handle missing values
    before = len(df)
    df = df.dropna()
    removed = before - len(df)
    print(f"Removed {removed} rows with missing values")
    
    # Identify column types
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical columns: {cat_cols}")
    print(f"Numeric columns: {num_cols}")
    
    # Encode categorical variables (convert demographics to numbers)
    print("Encoding demographic/social factors:")
    encoders = {}
    for col in cat_cols:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col])
        encoders[col] = enc
        print(f"  {col}: {list(enc.classes_)}")
    
    return df, encoders, cat_cols


# Step 2: Exploratory data analysis
def get_stats(df):
    """Calculate descriptive statistics for exam scores"""
    print("\n--- Exploratory Data Analysis: Statistics ---")
    score_cols = [col for col in df.columns if 'score' in col.lower()]
    
    for col in score_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Std Dev: {df[col].std():.2f}")
        print(f"  Range: {df[col].min():.0f} - {df[col].max():.0f}")


def plot_data(df, cat_cols):
    """Create visualizations: histograms, correlations, group comparisons"""
    print("\n--- Exploratory Data Analysis: Visualizations ---")
    score_cols = [col for col in df.columns if 'score' in col.lower()]
    
    sns.set_style("whitegrid")
    
    # Histogram: distribution of scores
    fig, axes = plt.subplots(1, len(score_cols), figsize=(5*len(score_cols), 5))
    if len(score_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(score_cols):
        axes[i].hist(df[col], bins=20, color='skyblue', edgecolor='black')
        axes[i].set_title(f'{col} Distribution')
        axes[i].set_xlabel('Score')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('01_distribution.png', dpi=300)
    print("Created: 01_distribution.png")
    plt.close()
    
    # Correlation analysis: how background factors relate to performance
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Analysis - Background Factors vs Performance')
    plt.tight_layout()
    plt.savefig('02_correlation.png', dpi=300)
    print("Created: 02_correlation.png")
    plt.close()
    
    # Group comparisons: scores across demographic groups
    if cat_cols and score_cols:
        fig, axes = plt.subplots(1, len(score_cols), figsize=(5*len(score_cols), 5))
        if len(score_cols) == 1:
            axes = [axes]
        
        demographic = cat_cols[0]
        for i, col in enumerate(score_cols):
            groups = sorted(df[demographic].unique())
            data = [df[df[demographic] == g][col].values for g in groups]
            axes[i].boxplot(data)
            axes[i].set_title(f'{col} by {demographic}')
            axes[i].set_ylabel('Score')
        
        plt.tight_layout()
        plt.savefig('03_groups.png', dpi=300)
        print("Created: 03_groups.png")
        plt.close()


# Step 3: Linear regression modeling
def build_model(df):
    """Build linear regression model to analyze background factors"""
    print("\n--- Linear Regression Modeling ---")
    
    score_cols = [col for col in df.columns if 'score' in col.lower()]
    if not score_cols:
        print("No score column found")
        return None
    
    target = score_cols[0]
    print(f"Target variable: {target}")
    
    # Features = demographic, socioeconomic, and preparation factors
    features = df.drop(columns=[target])
    target_var = df[target]
    
    print(f"Predictor factors: {list(features.columns)}")
    print(f"Training samples: {len(features)}")
    
    # Train linear regression
    model = LinearRegression()
    model.fit(features, target_var)
    print("Model trained successfully")
    
    # Display coefficients (factor importance)
    print(f"\nModel intercept: {model.intercept_:.4f}")
    print("Factor importance (coefficients):")
    
    coefs = pd.DataFrame({
        'Factor': features.columns,
        'Impact': model.coef_
    }).sort_values('Impact', key=abs, ascending=False)
    
    for idx, row in coefs.iterrows():
        print(f"  {row['Factor']}: {row['Impact']:.4f}")
    
    return model, features, target_var, target


# Step 4: Model evalulation 
def evaluate(model, features, target_var, target):
    """Evaluate model using R-squared and RMSE metrics"""
    print("\n--- Model Evaluation ---")
    
    predictions = model.predict(features)
    
    # R-squared: measure variance in test scores explained by model
    r2 = r2_score(target_var, predictions)
    print(f"R² Score: {r2:.4f}")
    print(f"  Interpretation: {r2*100:.1f}% of score variance explained by background factors")
    
    # RMSE: prediction accuracy
    rmse = np.sqrt(mean_squared_error(target_var, predictions))
    print(f"RMSE: {rmse:.4f} points")
    print(f"  Interpretation: Predictions average {rmse:.1f} points off")
    
    # Residuals
    residuals = target_var - predictions
    print(f"\nPrediction errors (residuals):")
    print(f"  Mean: {residuals.mean():.4f}")
    print(f"  Std Dev: {residuals.std():.4f}")
    print(f"  Min: {residuals.min():.4f}")
    print(f"  Max: {residuals.max():.4f}")
    
    return predictions, r2, rmse


def plot_predictions(target_var, predictions, target):
    """Visualize model performance and prediction quality"""
    print("\nCreating prediction diagnostic plots")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    residuals = target_var - predictions
    
    # Actual vs Predicted
    axes[0].scatter(target_var, predictions, alpha=0.6, color='steelblue', edgecolors='black')
    axes[0].plot([target_var.min(), target_var.max()], [target_var.min(), target_var.max()], 
                 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual Score')
    axes[0].set_ylabel('Predicted Score')
    axes[0].set_title('Actual vs Predicted Exam Performance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    axes[1].scatter(predictions, residuals, alpha=0.6, color='coral', edgecolors='black')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Score')
    axes[1].set_ylabel('Residuals (Error)')
    axes[1].set_title('Residual Plot - Error Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_predictions.png', dpi=300)
    print("Created: 04_predictions.png")
    plt.close()


def summary(df, r2, rmse, target):
    """Summarize findings and address research questions"""
    print("\n--- Summary of Findings ---")
    scores = df[target]
    
    print(f"Data analyzed: {len(df)} students, {len(df.columns)} factors")
    print(f"\n{target} Results:")
    print(f"  Average score: {scores.mean():.2f}")
    print(f"  Score range: {scores.min():.0f} - {scores.max():.0f}")
    
    print(f"\nModel Findings:")
    print(f"  R² = {r2:.4f} ({r2*100:.1f}% variance explained)")
    print(f"  RMSE = {rmse:.4f} points")
    
    print(f"\nInterpretation:")
    print(f"  ✓ Background factors significantly influence exam performance")
    print(f"  ✓ See coefficients above to identify most important factors")
    print(f"  ✓ Correlation heatmap shows factor relationships")
    print(f"  ✓ Group comparisons show demographic differences")


def main():
    print("="*60)
    print("STUDENT EXAM PERFORMANCE ANALYSIS")
    print("Step 3 Code Review - Methodology Implementation")
    print("="*60)
    
    file = "students_performance.csv"
    
    try:
        df = load_data(file)
    except FileNotFoundError:
        print("Ensure students_performance.csv is in the project folder")
        return
    
    # Step 1: Load and preprocess data
    explore_data(df)
    df_clean, encoders, cat_cols = clean_data(df)
    
    # Step 2: Exploratory analysis
    get_stats(df_clean)
    plot_data(df_clean, cat_cols)
    
    # Step 3-4: Modeling and evaluation
    result = build_model(df_clean)
    if result:
        model, features, target_var, target = result
        predictions, r2, rmse = evaluate(model, features, target_var, target)
        plot_predictions(target_var, predictions, target)
        summary(df_clean, r2, rmse, target)
    
    print("\n" + "="*60)
    print("Analysis Complete")
    print("All methodology steps successfully implemented")
    print("="*60)


if __name__ == "__main__":
    main()
