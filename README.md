# Housing Price Prediction

A machine learning model that predicts median house values based on California housing data. Uses scikit-learn with a Random Forest regressor and automated preprocessing pipeline.

## Overview

This tool trains a regression model on housing data or performs inference on new data. It handles data preprocessing, stratified sampling, and saves the trained model for reuse.

## Requirements

- Python 3.x
- pandas, numpy, scikit-learn, joblib

## Setup

Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

## Usage

### Training Mode

If `model.pkl` doesn't exist, the script automatically trains a new model:

```bash
python main.py
```

This will:
1. Load `housing.csv`
2. Create stratified train/test split based on income categories
3. Save test set to `input.csv`
4. Build preprocessing pipeline (imputation, scaling, one-hot encoding)
5. Train Random Forest model
6. Save model to `model.pkl` and pipeline to `pipeline.pkl`

### Inference Mode

If `model.pkl` exists, the script performs predictions:

```bash
python main.py
```

This will:
1. Load saved model and pipeline
2. Read data from `input.csv`
3. Apply preprocessing transformations
4. Generate predictions
5. Save results to `output.csv` with predicted `median_house_value`

## Data Format

### Input CSV Columns
- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- ocean_proximity (categorical)

### Output
Same columns plus predicted `median_house_value`

## How It Works

**Preprocessing Pipeline:**
- Numerical features: median imputation and standard scaling
- Categorical features: one-hot encoding with unknown value handling

**Model:**
- Random Forest Regressor with default parameters and random_state=42

**Data Split:**
- Stratified sampling based on income categories (5 bins)
- 80/20 train/test split
- Ensures representative distribution across income levels

## Files

- `main.py` - Training and inference script
- `housing.csv` - Original dataset
- `input.csv` - Test data (generated during training)
- `output.csv` - Predictions (generated during inference)
- `model.pkl` - Trained Random Forest model
- `pipeline.pkl` - Preprocessing pipeline

## Notes

- Delete `model.pkl` to retrain from scratch
- Model uses stratified splitting to maintain income distribution
- Pipeline handles missing values automatically
- Unknown categorical values handled gracefully during inference
