## License

This project is licensed under the MIT License for the source code.

The datasets used in this project are provided for academic/course purposes and are not redistributed under this license. If you use this repository, please make sure that you have permission to access and use the original data files.

# Real Estate Price Prediction

This repository contains a machine learning project for predicting real estate property prices using the **Zingat 2020** dataset. The project was developed for the **DS 540 / ML1** course at Özyeğin University.

The task is treated as a **supervised regression problem**, where the target variable is `price`. The main evaluation metrics are **Mean Squared Error (MSE)** and **Mean Absolute Percentage Error (MAPE)**.

---

## Project Overview

Real estate prices are influenced by many factors such as location, property category, area, room structure, building characteristics, broker information, and listing text. This project builds a full machine learning pipeline to estimate property sale prices from structured listing data.

The pipeline includes:

- Exploratory Data Analysis (EDA)
- Data cleaning and parsing of mixed-format variables
- Feature engineering
- Train-validation split with stratification
- Preprocessing with `ColumnTransformer`
- Target transformation using `log1p(price)`
- Model comparison
- Hyperparameter tuning with `GridSearchCV`
- Neural Network modeling
- Category-specific modeling for difficult property types
- Final prediction generation for the test set

---

## Dataset

The project uses two Zingat 2020 real estate listing files:

| File | Purpose | Shape |
|---|---:|---:|
| `zingat2020_1.csv` | Training data | 50,000 rows × 29 columns |
| `zingat2020_2.csv` | Test data | 50,000 rows × 29 columns |

The dataset includes property-level information such as:

- Location: `city`, `district`, `quarter`, `address`, coordinates
- Property details: `category`, `type`, `room`, `area`, `age`, `floor`, `total_floor`
- Building details: `heating_type`, `building_type`, `usage_status`
- Financial fields: `price`, `aidat`, `rentalIncome`
- Text fields: `title`, `description`
- Broker information

The target variable is:

```text
price
```

---

## Data Challenges

The raw dataset contains several real-world data problems:

- Highly skewed property prices
- Extremely high and extremely low price outliers
- Missing values in many columns
- Numerical values stored as text
- Turkish number formatting
- Combined latitude-longitude coordinate strings
- High-cardinality categorical variables
- Abnormal listings such as unit-price, share-based, rental-related, or installment-based advertisements
- Different category distributions between train and test data

These issues make the task more difficult than a simple regression problem and require careful preprocessing.

---

## Methodology

### 1. Data Cleaning

A single cleaning function is applied to both train and test datasets to keep preprocessing consistent. The main cleaning steps include:

- Dropping non-predictive columns such as IDs, URLs, contact fields, and empty columns
- Standardizing category labels
- Splitting coordinate strings into `latitude` and `longitude`
- Extracting broker company information
- Removing near-empty columns
- Parsing text-formatted numerical columns
- Converting floor, age, bathroom, and room information into usable numerical variables
- Removing impossible price outliers from the training set only

After cleaning, the training dataset contains **49,906 rows**.

### 2. Feature Engineering

Several domain-specific features are created, including:

- `log_price`
- `log_area`
- `log_area_sq`
- `room_count`
- `living_room_count`
- `total_room_count`
- `area_per_room`
- `floor_ratio`
- `floor_is_top`
- `age_sq`
- `is_land`
- `has_rental_income`
- Missing-value indicator features
- Text length features
- Location-combination features such as `city_district` and `district_quarter`

Keyword-based binary features are also extracted from listing text, such as:

- `has_metro`
- `has_sea`
- `has_parking`
- `has_pool`
- `has_site`
- `has_furnished`
- `has_new`
- `has_investment`
- `has_credit`
- `has_central`

Additional abnormal-listing flags are created to identify problematic advertisements:

- `is_unit_price_ad`
- `is_rental_text`
- `is_installment_text`
- `is_kat_karsiligi`
- `is_share_text`
- `is_bank_sale`
- `is_swap_text`

### 3. Preprocessing

Preprocessing is handled with a scikit-learn `ColumnTransformer`:

| Feature Type | Processing |
|---|---|
| Numerical features | Median imputation + standard scaling |
| Categorical features | Constant imputation + one-hot encoding |
| Missing indicators | Constant imputation + passthrough |

High-cardinality categorical variables are reduced by keeping the most frequent categories and grouping rare values as `Other`.

### 4. Target Transformation

Since the raw price distribution is highly right-skewed, the model is trained on:

```python
log_price = np.log1p(price)
```

Predictions are converted back to Turkish Lira using:

```python
predicted_price = np.expm1(predicted_log_price)
```

This makes the modeling process more stable and helps the models learn relative price differences more effectively.

---

## Models Used

The following regression models are trained and compared:

| Model | Purpose |
|---|---|
| Linear Regression | Simple baseline model |
| Decision Tree | Nonlinear baseline |
| Random Forest | Bagging-based tree ensemble |
| Gradient Boosting | Boosting-based model |
| XGBoost | Advanced boosting model |
| LightGBM | Efficient boosting model for tabular data |
| CatBoost | Boosting model suitable for categorical-heavy data |
| Neural Network / MLPRegressor | Required neural network model |

All models are evaluated using the same preprocessing pipeline.

---

## Hyperparameter Tuning

Hyperparameter tuning is performed with `GridSearchCV` using 3-fold cross-validation.

Tuned models include:

- Decision Tree
- Random Forest
- Gradient Boosting
- Neural Network
- LightGBM

The final global model is selected based on a balanced comparison of:

- Validation MSE
- Validation MAPE
- Validation Median APE

---

## Final Model Selection

The selected final global model is:

```text
Tuned LightGBM
```

The final model was selected because it achieved competitive performance on the required metrics and the best validation Median APE, which is more robust to extreme percentage errors.

### Validation Performance

| Model | Validation MSE | Validation MAPE | Validation Median APE |
|---|---:|---:|---:|
| Tuned LightGBM | 1.3326 × 10¹⁵ | 54.90% | 19.44% |
| Tuned Neural Network | 1.3316 × 10¹⁵ | 53.87% | 23.46% |
| Random Forest | 1.3389 × 10¹⁵ | 51.27% | 22.40% |

Although the Tuned Neural Network achieved the lowest validation MSE and Random Forest achieved the lowest validation MAPE, Tuned LightGBM was selected as the most stable overall model.

---

## Category-Specific Modeling

Because property categories behave differently, separate models are tested for categories with enough observations.

| Category | Decision |
|---|---|
| Konut | Category-specific model approved |
| Arsa | Category-specific model approved |
| Ticari | Category-specific model not approved |
| Turizm | Not enough data for a reliable separate model |

In the final prediction stage:

- Konut listings use the Konut-specific model
- Arsa listings use the Arsa-specific model
- Ticari and Turizm listings use the global Tuned LightGBM model

---

## Final Test Results

Final predictions are generated for all 50,000 rows in `zingat2020_2.csv`.

| Metric | Value |
|---|---:|
| MSE | 287,268,963,993,281.19 |
| RMSE | 16,949,010.71 TRY |
| MAE | 949,344.93 TRY |
| MAPE | 343.07% |
| Median APE | 24.25% |

The MAPE is high because a small number of very low-price abnormal listings create extremely large percentage errors. For this reason, Median APE is also reported as a more robust indicator of typical prediction performance.

---

## Category-Based Test Performance

| Category | Count | Median Actual Price | Median Predicted Price | Mean APE | Median APE |
|---|---:|---:|---:|---:|---:|
| Konut | 33,507 | 375,000 | 375,566 | 30.19% | 18.10% |
| Arsa | 12,292 | 360,000 | 354,136 | 1230.48% | 51.25% |
| Ticari | 4,114 | 850,000 | 793,846 | 241.39% | 42.79% |
| Turizm | 87 | 5,800,000 | 4,180,945 | 284.15% | 66.57% |

The model performs best on **Konut** listings. The most difficult category is **Arsa**, mainly because many land listings contain abnormal price structures such as unit prices or share-based sale information.

---

## Key Findings

- Real estate price prediction is strongly affected by data quality.
- Log transformation improves the target distribution and stabilizes model training.
- Tree-based ensemble models perform strongly on this tabular dataset.
- The Neural Network model is competitive but not selected as the final model.
- MSE is dominated by expensive outliers.
- MAPE is strongly affected by very low actual prices.
- Median APE gives a more realistic view of typical prediction performance.
- Residential listings are predicted much better than land, commercial, and tourism listings.

---

## Repository Structure

```text
real-estate-price-prediction/
│
├── real_estate_price_pred.ipynb
│
├── zingat2020_1.csv
├── zingat2020_2.csv
│
├── zingat2020_predictions.csv
├── zingat2020_predictions_detailed.csv
│
└── README.md
```

---

## Output Files

| File | Description |
|---|---|
| `zingat2020_predictions.csv` | Final predicted prices for the test set |
| `zingat2020_predictions_detailed.csv` | Final predictions with actual prices and APE values |
| `EzgiDonmez.ipynb` | Full notebook with code, outputs, models, and analysis |
| `EzgiDonmez.docx` | Written project report |

The main submission prediction file contains:

```text
externalId
Predicted Price
```

The detailed prediction file contains:

```text
externalId
Predicted Price
Actual Price
APE (%)
```

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ezgi-donmez/real-estate-price-prediction.git
cd real-estate-price-prediction
```

### 2. Create a virtual environment

Using conda:

```bash
conda create -n real-estate-price python=3.10
conda activate real-estate-price
```

Or using venv:

```bash
python -m venv real-estate-price-env
source real-estate-price-env/bin/activate
```

For Windows PowerShell:

```bash
real-estate-price-env\Scripts\Activate.ps1
```

### 3. Install required packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost jupyter
```

### 4. Open the notebook

```bash
jupyter notebook EzgiDonmez.ipynb
```

### 5. Run all cells

Run the notebook from top to bottom. The notebook will:

1. Load train and test data
2. Perform EDA
3. Clean and preprocess the data
4. Train baseline models
5. Train and evaluate the Neural Network
6. Tune selected models
7. Select the final model
8. Train category-specific models
9. Generate final test predictions
10. Save prediction files

---

## Main Libraries

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
lightgbm
catboost
jupyter
```

---

## Limitations

This project has some limitations:

- Some listings may not represent normal full sale prices.
- Very low actual prices make MAPE unstable.
- Very expensive properties strongly affect MSE and RMSE.
- Arsa, Ticari, and Turizm listings are harder to model than Konut listings.
- Text features are based on keyword flags rather than advanced NLP methods.
- Location effects could be modeled more deeply with target encoding or clustering.

---

## Future Work

Possible improvements include:

- Separating abnormal land listings from normal sale listings
- Using stronger location-based features
- Applying target encoding for high-cardinality categorical variables
- Testing wider hyperparameter tuning with `RandomizedSearchCV`
- Improving text feature extraction with NLP methods
- Building separate specialized models for more property types
- Testing additional robust evaluation metrics

---

## Author

**Ezgi Dönmez**  
MSc Data Science  
Özyeğin University  

---

## Course Context

This project was prepared for a machine learning course project focused on real estate price prediction. The required evaluation metrics are MSE and MAPE, and the project includes a complete ML pipeline and a Neural Network model.
