# Machine Learning Model Documentation

## Function: `train_multi_target_model(X_train, y_train)`

### Purpose
Trains a Multi-Output XGBoost Regressor with native categorical feature support. This model is suitable for regression tasks where there are multiple target variables (`y_train`) to predict simultaneously.

---

### Function Definition

```python
def train_multi_target_model(X_train, y_train):
    """
    Trains a Multi-Output XGBoost Regressor with categorical support enabled.
    """
    print("Initializing and training Multi-Output XGBoost Regressor...")
    
    # 1. Define the base estimator (XGBoost)
    base_estimator = XGBRegressor(
        n_estimators=100,             
        max_depth=7,                  
        learning_rate=0.1,            
        objective='reg:squarederror', 
        n_jobs=-1,                    
        random_state=RANDOM_SEED,
        tree_method='hist',           
        # **CRITICAL FIX: Enable native categorical feature handling**
        enable_categorical=True        
    )
    
    # 2. Wrap the base estimator for Multi-Output prediction
    model = MultiOutputRegressor(base_estimator)
    
    # 3. Train the model
    # Note: X_train must contain the columns listed in the error as 'category' dtype
    model.fit(X_train, y_train) 
    
    print("Training complete.")
    return model
```

---

### Key Steps

1. **Base Estimator:**  
   Utilizes `XGBRegressor` from XGBoost as the base estimator, configured specifically for regression with support for categorical features via `enable_categorical=True`.

2. **Multi-Output Prediction:**  
   The base estimator is wrapped using `MultiOutputRegressor` from scikit-learn, enabling prediction of multiple target variables.

3. **Training:**  
   - Model is trained on `X_train` and `y_train`.
   - **Important:** If there are categorical features, they must be of `category` dtype in `X_train` for native categorical handling to work.

---

### Parameters

- **X_train** (`pd.DataFrame`):  
  Training features, with categorical columns as type `category` where appropriate.

- **y_train** (`pd.DataFrame` or `np.ndarray`):  
  Target variables for regression. Must have the same number of rows as `X_train`.

---

### Returns

- **model** (`MultiOutputRegressor`):  
  The trained multi-output regressor model, ready for prediction on new data.

---

### Notes

- This approach leverages XGBoost’s ability to natively handle categorical data, improving performance and reducing the need for manual encoding.
- Ensure `xgboost` and `scikit-learn` (`sklearn`) are updated to versions that support `enable_categorical` in `XGBRegressor`.
- Set relevant columns in `X_train` to the `category` type (e.g., `X_train['feature'] = X_train['feature'].astype('category')`).

---

### Example Usage

```python
# Ensure categorical columns are properly set
X_train['cat_feature'] = X_train['cat_feature'].astype('category')
model = train_multi_target_model(X_train, y_train)
```

---

### References

- [XGBoost Documentation: Categorical Data](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)
- [scikit-learn: MultiOutputRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)