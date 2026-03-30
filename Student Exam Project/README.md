# Student Exam Performance Analysis

**Team:** Hannah Repuyan, Minwoo Choi, Victor Gutierrez  
**Course:** IS 392 Section 452  
**Date:** March 2026

## Overview

Analyzes how demographic, socioeconomic, and educational factors influence exam performance using linear regression and exploratory data analysis.

**Research Questions:**
1. How does demographic information influence exam performance?
2. Is parental education level related to exam performance?
3. Is there a correlation between test preparation and exam scores?

## Setup

1. **Download dataset** from [Kaggle](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
2. Save as `students_performance.csv` in project folder
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python Exam_proj.py`

## What the Code Does

**Step 1: Data Preprocessing**
- Removes missing values
- Encodes demographic categories to numbers
- Prepares data for analysis

**Step 2: Exploratory Analysis**
- Calculates mean, median, std dev of exam scores
- Creates histograms showing score distributions
- Builds correlation heatmap (background factors vs performance)
- Creates boxplots comparing scores across demographic groups

**Step 3: Linear Regression Model**
- Builds model predicting exam scores from background factors
- Shows coefficients for each factor's impact

**Step 4: Model Evaluation**
- R² Score: percentage of variance explained
- RMSE: average prediction error in points
- Identifies most important factors

## Output Files

- `01_distribution.png` - Score distributions
- `02_correlation.png` - Correlation heatmap
- `03_groups.png` - Comparison by demographic groups
- `04_predictions.png` - Model accuracy plots

## Requirements

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Known Issues & Next Steps

**Current Limitations:**
- Linear regression assumes linear relationships (may not capture complex patterns)
- Small dataset size may limit model accuracy
- Categorical encoding treats all categories ordinally
- No cross-validation or test set splitting

**Next Steps for Improvement:**
- Test additional models (Random Forest, Gradient Boosting)
- Implement k-fold cross-validation
- Try polynomial features or interaction terms
- Perform outlier detection and analysis
- Use one-hot encoding for categorical variables
- Add regularization (Ridge/Lasso) to prevent overfitting

## Notes

- Code runs end-to-end without manual intervention
- See console output for statistics and model results
- Linear regression chosen for interpretability and simplicity
- Label encoding used for categorical variables
