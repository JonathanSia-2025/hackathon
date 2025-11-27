# Flight Delay Prediction Model: Feature Preprocessing Report

This document outlines the preprocessing steps applied to the raw flight data to generate the 48 final features used to train the XGBoost model. This ensures that any new data being fed into the inference pipeline exactly matches the feature set and structure the model was trained on.

## 1. Feature Categories and Transformations

The final feature set can be divided into three main categories based on the transformation applied:

| **Category** | **Transformation Method** | **Original Raw Feature(s)** | **Example Final Feature(s)** | **Purpose** |
| --- | --- | --- | --- | --- |
| **I. Temporal / Cyclical** | Sine/Cosine Transformation | MONTH, DAY\_OF\_WEEK, CRS\_DEP\_TIME | MONTH\_SIN, DEP\_HOUR\_COS | Handles the cyclical nature of time (e.g., December is close to January). |
| **II. Categorical / Nominal** | One-Hot Encoding | OPERATING\_AIRLINE | AIRLINE\_AA, AIRLINE\_MQ | Converts text labels into a numerical format suitable for the model. |
| **III. Direct / Numerical** | Retained / Minor Cleaning | DISTANCE, DEP\_DELAY, ORIGIN\_STATE | FLIGHT\_DISTANCE, DEP\_DELAY | Features used directly after cleaning, renaming, or without explicit transformation. |

## 2. Detailed Preprocessing Methods

### A. Temporal and Cyclical Encoding

Time-based features, such as the month or time of day, are inherently cyclical (i.e., the time 23:59 is close to 00:01, and December is close to January). Standard integer encoding (1 for January, 12 for December) would imply that December is far from January, which is inaccurate.

To address this, we use **Sine and Cosine transformations** to map the time features onto a 2D circle.

| **Original Feature** | **Transformation Logic** | **Final Features** |
| --- | --- | --- |
| **Month** (1-12) | $\text{Sin} = \sin(2\pi \cdot \frac{\text{Month}}{12})$, $\text{Cos} = \cos(2\pi \cdot \frac{\text{Month}}{12})$ | MONTH\_SIN, MONTH\_COS |
| **Day of Week** (1-7) | $\text{Sin} = \sin(2\pi \cdot \frac{\text{Day}}{7})$, $\text{Cos} = \cos(2\pi \cdot \frac{\text{Day}}{7})$ | DAY\_OF\_WEEK\_SIN, DAY\_OF\_WEEK\_COS |
| **Departure Time** (Hour/Minute) | Time is split into hours (0-23) and minutes (0-59), and each is converted using the sine/cosine formula relative to its cycle size. | DEP\_HOUR\_SIN, DEP\_HOUR\_COS, DEP\_MIN\_SIN, DEP\_MIN\_COS |

### B. Categorical One-Hot Encoding (Airlines)

The raw data contains the OPERATING\_AIRLINE (or OP\_UNIQUE\_CARRIER) column as text identifiers (e.g., 'AA', 'DL', 'MQ'). Since XGBoost requires numerical input, this feature was transformed using **One-Hot Encoding (OHE)**.

* **Method:** For every unique airline code encountered in the training data, a new binary column was created (e.g., AIRLINE\_AA).
* **Result:** The final 26 features starting with AIRLINE\_ (e.g., AIRLINE\_9E through AIRLINE\_ZW) represent all known operating carriers. For any given flight, exactly one of these columns will have the value 1, and the rest will be 0.
* **Alignment Criticality:** It is absolutely essential that the inference data contains **all 26 of these binary columns in the exact same order**, even if a new flight belongs to an airline not present in that specific batch (in which case the column for that airline will be 1, and any missing known airline columns will be padded with 0).

### C. Direct and Numerical Features

These columns were used directly by the model, though several required minor cleanup (such as handling missing values or ensuring correct data types).

| **Final Feature Name** | **Original Source** | **Preprocessing Notes** |
| --- | --- | --- |
| DEP\_DELAY | Departure Delay in minutes | This is a crucial feature, as current delay is highly predictive of future, granular delays. Retained as a numerical feature. |
| FLIGHT\_DISTANCE | Distance in miles | Retained as a numerical feature. *Note: If a log transformation (DISTANCE\_LOG) was used in training, it must be added here, but the raw distance is retained in this final feature list.* |
| DISTANCE\_GROUP\_ID | Binning of distance | Retained as a numerical feature. |
| SCHEDULED\_FLIGHT\_DURATION | CRS\_ELAPSED\_TIME | Retained as numerical input. |
| AIR\_TIME | Actual flight time | Retained as numerical input. |
| **Location Codes** | Various raw columns | ORIGIN\_AIRPORT\_CODE, ORIGIN\_STATE, DEST\_AIRPORT\_CODE, DEST\_STATE, MARKETING\_AIRLINE, DEPARTURE\_BLOCK, ARRIVAL\_BLOCK are treated as discrete categorical features (XGBoost can often handle these as integers or native categories without explicit OHE). |
| DAY\_OF\_MONTH | Retained. | Numerical feature indicating the day of the month (1-31). |

### D. Data Filtering and Leakage Prevention

**1. Filtering Cancelled and Diverted Flights**

During the initial data preparation phase, rows representing cancelled or diverted flights were removed from the training data.

* **Filtering Logic:** df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]
* **Reasoning:** The objective of this regression model is to predict the *duration* of granular delays (Carrier, Security, Weather) for flights that are successfully completed. Flights that are cancelled or diverted represent a fundamentally different outcome (a binary/categorical prediction) and are typically addressed by separate classification models. Filtering these records ensures the regression model focuses exclusively on learning patterns related to delay duration for flights that are **in progress or completed**, thus preventing the training data from being skewed by non-delay outcomes. The columns CANCELLED and DIVERTED are dropped immediately after filtering is complete.

**2. Prevention of Data Leakage**

Data leakage is a phenomenon where information about the target variable (the outcome we are trying to predict) is unintentionally included in the features. This leads to artificially high performance during training which cannot be replicated during real-world inference.

The following columns were dropped during preprocessing to prevent data leakage because the information they contain is **only known after the flight has already arrived or completed its journey**:

| **Leakage Column (Example)** | **Why it causes Leakage** | **Action Taken** |
| --- | --- | --- |
| **Actual Arrival Time** (ARR\_TIME) | Knowing the actual arrival time essentially reveals the total delay (ARR\_DELAY), which is highly correlated with the target granular delay components. | Dropped. We use the *Scheduled* Arrival Time (CRS\_ARR\_TIME) instead. |
| **Arrival Delay** (ARR\_DELAY) | This is the metric that captures the end result of all delays (including the target variables). Including it would make the prediction trivial. | Dropped. |
| **Actual Wheels On/Off Times** | These temporal markers confirm the flight's status and final duration, which is future knowledge at the time of prediction. | Dropped. |

**Crucial Exception: Departure Delay (DEP\_DELAY)**

The column **DEP\_DELAY** was intentionally *kept* as a feature. The assumption for this model's deployment is that the prediction of granular delays occurs after the aircraft has pushed back and the initial departure delay is known. Since the departure delay is available *before* the flight has finished, it is a valid, highly predictive feature for predicting the subsequent, granular delay causes.
