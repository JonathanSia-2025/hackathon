import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from tkinter import Tk, filedialog # For simulating file upload

# =============================================================
# CONFIGURATION & CONTEXT (MUST MATCH TRAINING DATA)
# =============================================================

# --- Input Files ---
MODEL_FILENAME = 'xgb_flight_delayfactors_model_trained.joblib'
# Placeholder for testing, replace with user upload logic in a real UI
SCHEDULE_CSV_PATH = 'synthetic_factordelay_1500.csv' 

# --- Context (Inferred from training data—used for filtering and thresholds) ---
# NOTE: These values should ideally be calculated from your full y_train distribution.
HIST_MEAN_DELAY = 5.0 # Placeholder for mean historical delay (minutes)
HIST_STD_DELAY = 20.0 # Placeholder for std dev historical delay (minutes)
RISK_NEG_K = 3.0 # Factor for defining a "large" early arrival (e.g., 3 standard deviations early)

# --- Define the 52+ feature names used by the model ---
FEATURE_COLS = ['arr_del15', 'arr_cancelled', 'arr_diverted', 'arr_delay', 'carrier_ct_RATE',
                'weather_ct_RATE', 'nas_ct_RATE', 'security_ct_RATE', 'late_aircraft_ct_RATE',
                'month_sin', 'month_cos', 'time_id', 'AIRLINE_9E', 'AIRLINE_9K', 'AIRLINE_AA',
                'AIRLINE_AQ', 'AIRLINE_AS', 'AIRLINE_AX', 'AIRLINE_B6', 'AIRLINE_C5',
                'AIRLINE_CO', 'AIRLINE_CP', 'AIRLINE_DH', 'AIRLINE_DL', 'AIRLINE_EM', 
                'AIRLINE_EV', 'AIRLINE_F9', 'AIRLINE_FL', 'AIRLINE_G4', 'AIRLINE_G7',
                'AIRLINE_HA', 'AIRLINE_HP', 'AIRLINE_KS', 'AIRLINE_MQ', 'AIRLINE_NK', 
                'AIRLINE_NW', 'AIRLINE_OH', 'AIRLINE_OO', 'AIRLINE_PT', 'AIRLINE_QX', 
                'AIRLINE_RU', 'AIRLINE_TZ', 'AIRLINE_UA', 'AIRLINE_US', 'AIRLINE_VX', 
                'AIRLINE_WN', 'AIRLINE_XE', 'AIRLINE_YV', 'AIRLINE_YX', 'AIRLINE_ZW', 
                'airport_encoded']

# =============================================================
# CORE FUNCTIONS
# =============================================================

def load_model(filename: str):
    """Loads the trained XGBoost model."""
    try:
        model = joblib.load(filename)
        return model
    except FileNotFoundError:
        raise SystemExit(f"Error: Model file '{filename}' not found. Ensure it is in the script directory.")
    except Exception as e:
        raise SystemExit(f"Error loading model: {e}")

def load_artifacts():
    """Load encoder, feature column order, and target column names saved during training."""
    artifacts = {}
    try:
        artifacts['ordinal_encoder'] = joblib.load('ordinal_encoder.joblib')
    except Exception:
        artifacts['ordinal_encoder'] = None

    try:
        artifacts['feature_cols'] = joblib.load('feature_cols.joblib')
    except Exception:
        artifacts['feature_cols'] = None

    try:
        artifacts['target_cols'] = joblib.load('target_cols.joblib')
    except Exception:
        artifacts['target_cols'] = None

    return artifacts

def preprocess_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstructs the 52+ features required by the trained model and adds 
    necessary display columns (ROUTE, AIRLINE, DATE_OF_FLIGHT).
    ⚠️ CRITICAL: This is a placeholder. You must replace this 
    section with the EXACT 52-feature construction logic from your training script.
    """
    df_proc = df_raw.copy()
    df_proc.columns = df_proc.columns.str.upper()

    # If MARKETING_AIRLINE is missing or empty, try to infer from one-hot AIRLINE_* columns
    if 'MARKETING_AIRLINE' not in df_proc.columns or df_proc['MARKETING_AIRLINE'].astype(str).str.strip().eq('').all():
        airline_cols = [c for c in df_proc.columns if c.startswith('AIRLINE_')]
        if airline_cols:
            def infer_airline(row):
                for c in airline_cols:
                    try:
                        val = row.get(c, 0)
                        if float(val) == 1.0:
                            return c.replace('AIRLINE_', '')
                    except Exception:
                        if str(row.get(c, '')).strip() in ('1', '1.0'):
                            return c.replace('AIRLINE_', '')
                return ''
            df_proc['MARKETING_AIRLINE'] = df_proc.apply(infer_airline, axis=1)
        else:
            df_proc['MARKETING_AIRLINE'] = ''

    # If ORIGIN/DEST airport codes are missing, attempt to recover from AIRPORT_ENCODED
    if ('ORIGIN_AIRPORT_CODE' not in df_proc.columns or df_proc['ORIGIN_AIRPORT_CODE'].astype(str).str.strip().eq('').all()) or \
       ('DEST_AIRPORT_CODE' not in df_proc.columns or df_proc['DEST_AIRPORT_CODE'].astype(str).str.strip().eq('').all()):
        if 'AIRPORT_ENCODED' in df_proc.columns:
            codes = ['SFO', 'LAX', 'JFK', 'ATL', 'ORD', 'MIA', 'DFW', 'BOS']
            def code_from_enc(val):
                try:
                    n = int(float(val))
                    return codes[n % len(codes)]
                except Exception:
                    return ''
            df_proc['ORIGIN_AIRPORT_CODE'] = df_proc.get('ORIGIN_AIRPORT_CODE', pd.Series(['']*len(df_proc))).astype(str)
            df_proc['DEST_AIRPORT_CODE'] = df_proc.get('DEST_AIRPORT_CODE', pd.Series(['']*len(df_proc))).astype(str)
            # Fill missing ones from encoded
            mask_o = df_proc['ORIGIN_AIRPORT_CODE'].str.strip() == ''
            mask_d = df_proc['DEST_AIRPORT_CODE'].str.strip() == ''
            if mask_o.any():
                df_proc.loc[mask_o, 'ORIGIN_AIRPORT_CODE'] = df_proc.loc[mask_o, 'AIRPORT_ENCODED'].apply(code_from_enc)
            if mask_d.any():
                # offset destination to reduce chance of equality
                def dest_from_enc(v):
                    try:
                        n = int(float(v))
                        return codes[(n+1) % len(codes)]
                    except Exception:
                        return ''
                df_proc.loc[mask_d, 'DEST_AIRPORT_CODE'] = df_proc.loc[mask_d, 'AIRPORT_ENCODED'].apply(dest_from_enc)
        else:
            df_proc['ORIGIN_AIRPORT_CODE'] = df_proc.get('ORIGIN_AIRPORT_CODE', pd.Series(['']*len(df_proc))).astype(str)
            df_proc['DEST_AIRPORT_CODE'] = df_proc.get('DEST_AIRPORT_CODE', pd.Series(['']*len(df_proc))).astype(str)

    # --- Feature Reconstruction Placeholder ---
    
    # 1. Ensure required feature columns are present (filling missing with 0 for safety)
    for col in FEATURE_COLS:
        if col not in df_proc.columns:
            df_proc[col] = 0.0

    # 2. Reconstruct categorical/route columns for filtering and display
    
    # Check for existence and apply aggressive cleaning/upper-casing
    airline_col = df_proc.get('MARKETING_AIRLINE', pd.Series(''))
    orig_col = df_proc.get('ORIGIN_AIRPORT_CODE', pd.Series(''))
    dest_col = df_proc.get('DEST_AIRPORT_CODE', pd.Series(''))

    df_proc['AIRLINE'] = airline_col.astype(str).str.strip().str.upper()
    df_proc['ORIGIN_AIRPORT_CODE'] = orig_col.astype(str).str.strip().str.upper()
    df_proc['DEST_AIRPORT_CODE'] = dest_col.astype(str).str.strip().str.upper()

    # Reconstruct ROUTE
    df_proc['ROUTE'] = df_proc['ORIGIN_AIRPORT_CODE'] + "-" + df_proc['DEST_AIRPORT_CODE']
    
    # --- CRITICAL FIX: Filter out problematic codes immediately ---
    
    # Identify placeholder values that result from NaNs or mixed types being converted to string
    PLACEHOLDERS = {'0.0', 'UNK', 'UNKNOWN-ROUTE', 'NAN', ''}
    
    # Check 1: Route components must not be placeholders
    valid_mask_placeholders = (df_proc['ORIGIN_AIRPORT_CODE'].isin(PLACEHOLDERS) == False) & \
                              (df_proc['DEST_AIRPORT_CODE'].isin(PLACEHOLDERS) == False) & \
                              (df_proc['AIRLINE'].isin(PLACEHOLDERS) == False)
    
    # Check 2: Airport codes must be 3 characters long (filter out the numeric '0.0', '1.0', etc.)
    valid_mask_length = (df_proc['ORIGIN_AIRPORT_CODE'].str.len() == 3) & \
                        (df_proc['DEST_AIRPORT_CODE'].str.len() == 3)

    # Check 3: Airport codes must contain only letters (to eliminate accidental numeric/corrupted entries)
    valid_mask_alpha = (df_proc['ORIGIN_AIRPORT_CODE'].str.isalpha()) & \
                       (df_proc['DEST_AIRPORT_CODE'].str.isalpha())

    df_proc = df_proc[valid_mask_placeholders & valid_mask_length & valid_mask_alpha].reset_index(drop=True)
    
    if df_proc.empty:
        raise ValueError("All rows filtered out. Check CSV columns for valid 3-letter IATA Airport/Airline codes.")

    # 3. Add a date column placeholder for display/saving (using available date parts)
    # Ensure YEAR/MONTH/DAY_OF_MONTH are Series of correct length (defaults when missing)
    def _as_series_or_default(df, colname, default):
        val = df.get(colname, None)
        if val is None:
            return pd.Series([default] * len(df), index=df.index)
        # If a scalar slipped through (not a Series), make a Series of that scalar
        if not isinstance(val, (pd.Series, pd.Index)):
            return pd.Series([val] * len(df), index=df.index)
        return val

    year_s = _as_series_or_default(df_proc, 'YEAR', 2024).astype(str)
    month_s = _as_series_or_default(df_proc, 'MONTH', 1).astype(str).str.zfill(2)
    day_s = _as_series_or_default(df_proc, 'DAY_OF_MONTH', 1).astype(str).str.zfill(2)

    df_proc['DATE_OF_FLIGHT'] = pd.to_datetime(year_s + '-' + month_s + '-' + day_s, errors='coerce')
    df_proc['DATE_OF_FLIGHT'] = df_proc['DATE_OF_FLIGHT'].fillna(pd.Timestamp.now().normalize())

    # 4. Finalize data types and order for prediction
    df_pred_ready = df_proc[FEATURE_COLS].copy()
    
    for col in FEATURE_COLS:
        if df_pred_ready[col].dtype == 'object':
            # Converting string features to numeric 0 if they failed to be encoded
            df_pred_ready[col] = pd.to_numeric(df_pred_ready[col], errors='coerce').fillna(0).astype(np.int32)
        elif df_pred_ready[col].dtype != 'float32':
             df_pred_ready[col] = df_pred_ready[col].astype('float32')
            
    # Add back columns needed for display/filtering/saving
    df_pred_ready['AIRLINE'] = df_proc['AIRLINE']
    df_pred_ready['ROUTE'] = df_proc['ROUTE']
    df_pred_ready['SCHEDULED_ARRIVAL_TIME'] = df_proc.get('SCHEDULED_ARRIVAL_TIME', 0)
    df_pred_ready['DATE_OF_FLIGHT'] = df_proc['DATE_OF_FLIGHT']
    df_pred_ready['ORIGIN_AIRPORT_CODE'] = df_proc['ORIGIN_AIRPORT_CODE']
    df_pred_ready['DEST_AIRPORT_CODE'] = df_proc['DEST_AIRPORT_CODE']
    
    return df_pred_ready


def generate_future_rows(df: pd.DataFrame, n_extra: int) -> pd.DataFrame:
    """Append `n_extra` future rows by copying the last row and incrementing the date fields.
    This keeps route/airline/origin/dest and scheduled times the same and updates YEAR/MONTH/DAY_OF_MONTH/DATE_OF_FLIGHT.
    """
    if n_extra <= 0:
        return df

    # Ensure DATE_OF_FLIGHT is datetime
    df = df.copy()
    df['DATE_OF_FLIGHT'] = pd.to_datetime(df['DATE_OF_FLIGHT'])

    last_row = df.iloc[-1]
    future_rows = []
    for i in range(1, n_extra + 1):
        new_row = last_row.copy()
        new_date = last_row['DATE_OF_FLIGHT'] + pd.Timedelta(days=i)
        new_row['DATE_OF_FLIGHT'] = new_date
        # Update simple date parts that may be used by the model or display
        new_row['YEAR'] = float(new_date.year)
        new_row['MONTH'] = float(new_date.month)
        new_row['DAY_OF_MONTH'] = float(new_date.day)
        # Ensure all model FEATURE_COLS are numeric and have consistent dtypes
        for f in FEATURE_COLS:
            if f in new_row.index:
                # If the value is non-numeric (e.g., 'SFO' left in ORIGIN_AIRPORT_CODE),
                # coerce to numeric and fill with 0. This keeps the matrix numeric for XGBoost.
                try:
                    # Try converting to float first
                    new_row[f] = float(new_row[f])
                except Exception:
                    # Coerce strings to numeric where possible, else 0.0
                    try:
                        coerced = pd.to_numeric(pd.Series([new_row[f]]), errors='coerce').iloc[0]
                        if pd.isna(coerced):
                            new_row[f] = 0.0
                        else:
                            new_row[f] = float(coerced)
                    except Exception:
                        new_row[f] = 0.0
        future_rows.append(new_row)

    if future_rows:
        df_extra = pd.DataFrame(future_rows)
        df_out = pd.concat([df, df_extra], ignore_index=True)
        return df_out
    return df

def interactive_filter(df: pd.DataFrame):
    """
    Prompts user for route prefix using dynamic examples, lists matches, 
    and returns filtered data based on user input for route and airline.
    """
    
    all_routes = sorted(df['ROUTE'].unique())
    
    # Extract unique airport codes for dynamic examples
    unique_origins = df['ORIGIN_AIRPORT_CODE'].unique()
    unique_dests = df['DEST_AIRPORT_CODE'].unique()
    
    # Filter out UNK and combine codes
    all_codes = np.unique(np.concatenate([unique_origins, unique_dests]))
    clean_codes = [c for c in all_codes if c != 'UNK' and c != 'UNKNOWN-ROUTE' and len(c) == 3] # Assume standard 3-letter codes
    
    # --- Robust Example Selection ---
    
    # 1. Try to get 3 random clean codes
    if len(clean_codes) >= 3:
        np.random.seed(42) 
        example_codes = np.random.choice(clean_codes, 3, replace=False)
    # 2. If < 3 codes, use what's available
    elif len(clean_codes) > 0:
        example_codes = clean_codes
    # 3. Fallback to an empty list (DO NOT USE HARDCODED SFO/LAX)
    else:
        example_codes = [] 
    
    print("\n--- 3a. Filter Routes ---")
    
    # --- Route Selection ---
    selected_route = None
    while selected_route is None:
        # 1. Prompt for prefix with dynamic examples
        example_str = ", ".join([f"'{c}'" for c in example_codes])
        
        # NOTE: If no examples are available, example_str is empty, prompting like: "Example: "
        query = input(f"Enter route prefix/suffix (Example: {example_str}): ").strip().upper()
        
        # --- Check for matches ---
        matches = [r for r in all_routes if query in r]
        
        if not query:
            # If user presses Enter (no query), show sample
            matches = all_routes[:50]
            print(f"\nShowing sample routes ({len(matches)}):")
        elif not matches:
            # If search fails, inform user and SHOW THE AVAILABLE LIST
            print(f"No routes matching prefix/suffix '{query}' found.")
            
            # --- CORRECTED FLOW: Show available routes and ask for exact route ---
            print("\nPlease review the sample routes below to enter the exact match:")
            # Display sample of available routes from the entire set
            print(f"Available sample routes: {', '.join(all_routes[:5])}")
            # If search fails, we skip the print-matches step, but proceed to the exact route input
            
        else:
            # If matches found, print them
            print(f"\nMatching Routes ({len(matches)} total, showing max 50):")
            for i, r in enumerate(matches[:50]):
                print(f"  {i+1}: {r}")

        # 2. Prompt for exact route selection (This prompt always runs if matches were found or query was empty)
        route_input = input("Enter the EXACT route (e.g., 'SFO-LAX') to forecast: ").strip().upper()
        
        if route_input in all_routes:
            selected_route = route_input
        else:
            print("Invalid input. Please enter an exact route from the list.")
            
    # Filter by selected route
    df_route_filtered = df[df['ROUTE'] == selected_route]
    valid_airlines = sorted(df_route_filtered['AIRLINE'].unique())
    valid_airlines = [a for a in valid_airlines if a.strip()] # Remove empty/nan airlines

    # --- Airline Selection ---
    print("\n--- 3b. Select Airline ---")
    print(f"Airlines flying {selected_route}: {valid_airlines}")
    
    selected_airline = None
    while selected_airline is None:
        airline_input = input("Choose EXACT airline code (e.g., 'UA') to forecast: ").strip().upper()

        if airline_input in valid_airlines:
            selected_airline = airline_input
            break
        print(f"\nInvalid airline '{airline_input}'. Please choose from: {valid_airlines}")

    print(f"\n✔ Selected: Airline '{selected_airline}' on Route '{selected_route}'.")
    
    return df_route_filtered[df_route_filtered['AIRLINE'] == selected_airline].reset_index(drop=True), selected_airline, selected_route


def flag_anomalies(df_pred: pd.DataFrame) -> pd.DataFrame:
    """Flags anomalies for multi-output factor predictions.

    Rules applied:
    - For each factor target (e.g., carrier_ct_RATE), any positive value (>0)
      is considered a risk and is flagged. Negative values are left unflagged.
    - Each positive value is categorized into LOW/MEDIUM/HIGH using minute thresholds.
    - Produces per-target boolean `RISK_<target>` and `RISK_LEVEL_<target>` columns,
      an overall `RISK` boolean, a `RISK_LEVEL_SUMMARY` (highest severity), and a
      concise `MESSAGE` suitable for terminal output and CSVs.
    """

    # Risk thresholds (minutes) - same as single-delay script
    LOW_THRESH = 15.0
    MED_THRESH = 60.0

    # Determine candidate target columns: prefer explicit trained_target_cols if present
    trained_targets = globals().get('trained_target_cols', None)
    if trained_targets:
        target_cols = [t for t in trained_targets if t in df_pred.columns]
    else:
        # Fallback: choose columns that look like factor predictions (end with _RATE or start with PRED_ or PREDICTED)
        target_cols = [c for c in df_pred.columns if c.endswith('_RATE') or c.startswith('PRED_') or c == 'PREDICTED_DELAY']

    # Ensure deterministic order for display
    target_cols = sorted(target_cols)

    severity_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}

    # Create per-target risk and level columns
    per_levels = []
    for t in target_cols:
        rc = f'RISK_{t}'
        rl = f'RISK_LEVEL_{t}'
        # Coerce column to numeric once for safer comparisons
        s = pd.to_numeric(df_pred[t], errors='coerce').fillna(0.0)
        # Flag positive values as risks; negatives remain False
        df_pred[rc] = s > 0
        # Default level blank
        df_pred[rl] = ''
        # Assign levels using explicit parenthesis to avoid operator precedence issues
        df_pred.loc[(s > 0) & (s <= LOW_THRESH), rl] = 'LOW'
        df_pred.loc[(s > LOW_THRESH) & (s <= MED_THRESH), rl] = 'MEDIUM'
        df_pred.loc[s > MED_THRESH, rl] = 'HIGH'
        per_levels.append(rl)

    # Overall RISK is True if any per-target risk is True
    per_risk_cols = [f'RISK_{t}' for t in target_cols]
    if per_risk_cols:
        df_pred['RISK'] = df_pred[per_risk_cols].any(axis=1)
    else:
        df_pred['RISK'] = False

    # Compute summary risk level: the highest severity among per-target levels
    def _level_to_score(s):
        return severity_map.get(s, 0)

    def _score_to_level(score):
        if score >= 3:
            return 'HIGH'
        if score == 2:
            return 'MEDIUM'
        if score == 1:
            return 'LOW'
        return ''

    def _compute_summary(row):
        best = 0
        for rl in per_levels:
            lvl = row.get(rl, '')
            s = _level_to_score(lvl)
            if s > best:
                best = s
        return _score_to_level(best)

    if per_levels:
        df_pred['RISK_LEVEL_SUMMARY'] = df_pred.apply(_compute_summary, axis=1)
    else:
        df_pred['RISK_LEVEL_SUMMARY'] = ''

    # Build concise per-row MESSAGE: short tokens for each target like 'c:+1.2(L)'
    short_names = {}
    for t in target_cols:
        short = t.replace('_ct_RATE', '').replace('_delay_RATE', '').replace('PREDICTED_DELAY', 'total')
        short_names[t] = short

    def _make_short_message(row):
        parts = []
        for t in target_cols:
            val = row.get(t, 0.0)
            try:
                v = float(val)
                tag = row.get(f'RISK_LEVEL_{t}', '')
                tag_short = ''
                if tag == 'LOW':
                    tag_short = 'L'
                elif tag == 'MEDIUM':
                    tag_short = 'M'
                elif tag == 'HIGH':
                    tag_short = 'H'
                else:
                    tag_short = '-'
                parts.append(f"{short_names[t]}:{v:+.1f}({tag_short})")
            except Exception:
                parts.append(f"{short_names[t]}:NA(-)")
        # Append overall summary
        summary = row.get('RISK_LEVEL_SUMMARY', '')
        overall = f"SUM:{summary}" if summary else "SUM:-"
        return ' | '.join(parts) + ' | ' + overall

    df_pred['MESSAGE'] = df_pred.apply(_make_short_message, axis=1)
    return df_pred


# =============================================================
# MAIN EXECUTION
# =============================================================

if __name__ == "__main__":
    
    # --- 1. Load Model ---
    model = load_model(MODEL_FILENAME)
    artifacts = load_artifacts()
    # If training saved feature_cols/target_cols, prefer them
    trained_feature_cols = artifacts.get('feature_cols')
    trained_target_cols = artifacts.get('target_cols')
    ord_enc = artifacts.get('ordinal_encoder')
    if trained_feature_cols is not None:
        FEATURE_ORDER = trained_feature_cols
    else:
        FEATURE_ORDER = FEATURE_COLS
    
    print("\nModel loaded successfully. Ready for prediction.")
    
    # --- 2. Load Scheduled CSV Data (Simulated Upload) ---
    print("\n--- 2. Loading Schedule Data ---")
    try:
        # Hide the Tkinter root window
        Tk().withdraw()

        # Let the user pick a file (defaults to Documents) — fallback to the hardcoded path
        file_path = filedialog.askopenfilename(initialdir=os.path.expanduser('~\\Documents'),
                                               title='Select schedule CSV',
                                               filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])

        if not file_path:
            # No selection: fall back to the default used for testing
            file_path = SCHEDULE_CSV_PATH
            print(f"No file selected. Falling back to default: {file_path}")

        # NOTE: Using dtype=str ensures the airport codes are read as clean strings, preventing the 0.0 error.
        # Enforcing NA values to be empty strings helps robustness.
        df_raw_schedule = pd.read_csv(file_path, dtype=str, keep_default_na=False, na_values=[''])
        print(f"Loaded {len(df_raw_schedule)} total raw schedule rows from {file_path}.")

    except Exception as e:
        print(f"Error loading schedule data: {e}")
        raise SystemExit(1)

    # --- Preprocess Data ---
    try:
        # This will now filter out rows where airport codes are not clean 3-letter strings
        df_processed = preprocess_and_engineer(df_raw_schedule)
    except ValueError as e:
        print(f"\nCRITICAL ERROR during preprocessing: {e}")
        print("Please check your CSV file for valid airport and airline codes.")
        raise SystemExit(1)
        
    # --- 3. Filter Data by User Input (Interactive) ---
    df_filtered, selected_airline, selected_route = interactive_filter(df_processed)

    if df_filtered.empty:
        print("\nNo rows matched the selected criteria after filtering.")
        raise SystemExit(1)
        
    # --- 4. Prompt for number of future DAYS to forecast ---
    while True:
        try:
            # The user wants predictions for future days (e.g., next 7 days).
            ndays = int(input("\nHow many future DAYS to forecast (e.g., 7): ").strip())
            if ndays <= 0:
                print("Please enter an integer greater than 0.")
                continue

            # Generate exactly `ndays` future rows starting after the last available date
            print(f"Generating {ndays} future day rows based on the last available schedule row.")
            df_filtered = generate_future_rows(df_filtered, ndays)
            # We'll predict only the generated future rows (the last `ndays` rows)
            days_to_forecast = ndays
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # --- 5. Predict Delay ---
    
    # 1. Prepare data: predict only the future rows we just generated (last `days_to_forecast` rows)

    # Build feature matrix using the exact feature order used in training
    # Match FEATURE_ORDER entries to df_filtered column names case-insensitively.
    matched_cols = []
    for f in FEATURE_ORDER:
        # find case-insensitive match in df_filtered
        candidates = [c for c in df_filtered.columns if c.lower() == str(f).lower()]
        if candidates:
            matched_cols.append(candidates[0])
        else:
            # If missing, create the column with zeros so model has consistent shape
            df_filtered[f] = 0.0
            matched_cols.append(f)
    X_new = df_filtered[matched_cols].copy()

    # If an OrdinalEncoder is available, detect which columns to encode by
    # comparing encoder.n_features_in_ with candidate object-like columns in FEATURE_ORDER
    if ord_enc is not None:
        # Infer candidate categorical columns in order
        candidate_cols = [c for c in FEATURE_ORDER if c in df_filtered.columns and df_filtered[c].dtype == 'object']
        # Fallback: also consider columns containing any non-numeric values
        if len(candidate_cols) < getattr(ord_enc, 'n_features_in_', 0):
            candidate_cols = [c for c in FEATURE_ORDER if c in df_filtered.columns and df_filtered[c].apply(lambda v: isinstance(v, str)).any()]
        # Trim or expand to match encoder's expected input count
        try:
            n_expected = ord_enc.n_features_in_
        except Exception:
            n_expected = len(candidate_cols)
        cat_cols = candidate_cols[:n_expected]

        if cat_cols:
            # Normalize text then transform
            X_new[cat_cols] = X_new[cat_cols].astype(str).apply(lambda s: s.str.strip().str.upper())
            try:
                X_new[cat_cols] = ord_enc.transform(X_new[cat_cols].astype(str))
            except Exception as e:
                print(f"Warning: OrdinalEncoder transform failed: {e}. Falling back to numeric coercion.")
                X_new = X_new.apply(pd.to_numeric, errors='coerce')
    else:
        # No encoder available — coerce everything to numeric (legacy behavior)
        X_new = X_new.apply(pd.to_numeric, errors='coerce')

    X_new = X_new.fillna(0).astype('float32')

    # Select only the last `days_to_forecast` rows as the target prediction set
    X_pred = X_new.tail(days_to_forecast).copy()

    # 2. Run prediction on those future rows (model may be multi-output)
    # Ensure X_pred columns match model's expected feature names (exact case-sensitive match)
    def _get_expected_feature_names(m):
        try:
            return m.get_booster().feature_names
        except Exception:
            try:
                # MultiOutputRegressor stores individual estimators
                ests = getattr(m, 'estimators_', None)
                if ests and len(ests) > 0:
                    return ests[0].get_booster().feature_names
            except Exception:
                return None
        return None

    expected_names = _get_expected_feature_names(model)
    if expected_names is not None:
        final_cols = []
        for en in expected_names:
            # case-insensitive match
            cand = [c for c in X_pred.columns if c.lower() == str(en).lower()]
            if cand:
                final_cols.append(cand[0])
            else:
                # create missing column filled with zeros
                X_pred[en] = 0.0
                final_cols.append(en)
        # Reorder and rename exactly to match expected names
        X_pred = X_pred[final_cols]
        X_pred.columns = expected_names

    predictions = model.predict(X_pred)

    # 3. Attach predictions back to the corresponding rows in df_filtered
    pred_idx = df_filtered.index[-days_to_forecast:]
    # If predictions is 1D, keep legacy 'PREDICTED_DELAY' column
    if getattr(predictions, 'ndim', 1) == 1:
        df_filtered.loc[pred_idx, 'PREDICTED_DELAY'] = predictions
    else:
        # Use trained_target_cols if available, otherwise create generic names
        if trained_target_cols is not None:
            tgt_names = trained_target_cols
        else:
            tgt_names = [f'PRED_{i}' for i in range(predictions.shape[1])]
        # Attach each column
        for i, name in enumerate(tgt_names):
            df_filtered.loc[pred_idx, name] = predictions[:, i]

    # 4. Flag anomalies across the predicted slice
    df_final = flag_anomalies(df_filtered)
    
    # --- 6. Display and Slice Results for Output ---
    
    # Now, slice the final output dataframe to the generated future rows (last `days_to_forecast` rows)
    df_output = df_final.tail(days_to_forecast).copy()

    # Improve readability: compute component sums, reconcile with total if present, round values
    # Identify predicted component columns (commonly end with '_delay_RATE') and component count columns ('_ct_RATE')
    pred_comp_cols = [c for c in df_output.columns if c.endswith('_delay_RATE')]
    obs_comp_cols = [c for c in df_output.columns if c.endswith('_ct_RATE')]

    # Determine total predicted column if present
    total_pred_col = 'PREDICTED_DELAY' if 'PREDICTED_DELAY' in df_output.columns else None

    # Compute component-predicted sum
    if pred_comp_cols:
        # Ensure components are numeric to prevent string concatenation issues
        df_output[pred_comp_cols] = df_output[pred_comp_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        df_output['comp_pred_sum'] = df_output[pred_comp_cols].sum(axis=1)
    else:
        df_output['comp_pred_sum'] = 0.0

    # If model provides a total, use it; otherwise derive total from component sum
    if total_pred_col:
        df_output['pred_total'] = df_output[total_pred_col].astype(float)
    else:
        df_output['pred_total'] = df_output['comp_pred_sum']

    # Reconciliation: if component sum > total, scale components proportionally to match total
    def reconcile_row(row):
        comp_sum = float(row['comp_pred_sum'])
        total = float(row['pred_total'])
        reconciled = False
        if comp_sum > 0 and total >= 0 and comp_sum > total:
            scale = total / comp_sum if comp_sum > 0 else 0.0
            for c in pred_comp_cols:
                row[c] = float(row[c]) * scale
            row['comp_pred_sum'] = total
            reconciled = True
        # If total is zero but components positive, keep components and set pred_total to comp_sum
        if total == 0 and comp_sum > 0:
            row['pred_total'] = comp_sum
        row['reconciled'] = reconciled
        return row

    if pred_comp_cols:
        df_output = df_output.apply(reconcile_row, axis=1)
    else:
        df_output['reconciled'] = False

    # Round numeric display columns to 1 decimal for readability
    round_cols = pred_comp_cols + obs_comp_cols + ['comp_pred_sum', 'pred_total']
    for c in round_cols:
        if c in df_output.columns:
            df_output[c] = pd.to_numeric(df_output[c], errors='coerce').round(1)

    print("\n===== FORECAST RESULTS =====")
    print(f"Airline: {selected_airline}, Route: {selected_route}")

    # Prepare concise display columns: date, scheduled arrival, predicted components, sums, message/risk
    display_pred = pred_comp_cols if pred_comp_cols else []
    output_cols_display = ['DATE_OF_FLIGHT', 'SCHEDULED_ARRIVAL_TIME'] + display_pred + ['comp_pred_sum', 'pred_total', 'reconciled', 'MESSAGE', 'RISK']
    output_cols_display = [c for c in output_cols_display if c in df_output.columns]

    # Truncate long MESSAGE for terminal display only
    df_display = df_output[output_cols_display].copy()
    if 'MESSAGE' in df_display.columns:
        df_display['MESSAGE'] = df_display['MESSAGE'].astype(str).apply(lambda s: (s[:120] + '...') if len(s) > 120 else s)

    # Build a tidy, fixed-column table for terminal output
    COMPONENT_ORDER = ['carrier_delay_RATE', 'weather_delay_RATE', 'nas_delay_RATE', 'security_delay_RATE', 'late_aircraft_delay_RATE']
    pretty_cols = ['DATE_OF_FLIGHT', 'SCHEDULED_ARRIVAL_TIME'] + [c for c in COMPONENT_ORDER if c in df_output.columns] + ['pred_total', 'RISK', 'MESSAGE']
    df_pretty = df_output[pretty_cols].copy()

    # Round numeric component columns
    for c in COMPONENT_ORDER + ['pred_total']:
        if c in df_pretty.columns:
            df_pretty[c] = pd.to_numeric(df_pretty[c], errors='coerce').round(1).fillna(0.0)

    # Shorten messages
    if 'MESSAGE' in df_pretty.columns:
        df_pretty['MESSAGE_SHORT'] = df_pretty['MESSAGE'].astype(str).apply(lambda s: (s[:100] + '...') if len(s) > 100 else s)
    else:
        df_pretty['MESSAGE_SHORT'] = ''

    # Format date and schedule for display
    def _fmt_date(x):
        try:
            return pd.to_datetime(x).strftime('%Y-%m-%d')
        except Exception:
            return str(x)

    df_pretty['DATE_OF_FLIGHT'] = df_pretty['DATE_OF_FLIGHT'].apply(_fmt_date)
    df_pretty['SCHEDULED_ARRIVAL_TIME'] = df_pretty['SCHEDULED_ARRIVAL_TIME'].fillna('').astype(str)

    # Final display columns and headers
    display_order = ['DATE_OF_FLIGHT', 'SCHEDULED_ARRIVAL_TIME'] + [c for c in COMPONENT_ORDER if c in df_pretty.columns] + ['pred_total', 'RISK', 'MESSAGE_SHORT']
    # Rename component headers to short names
    rename_map = { 'carrier_delay_RATE':'carrier', 'weather_delay_RATE':'weather', 'nas_delay_RATE':'nas', 'security_delay_RATE':'security', 'late_aircraft_delay_RATE':'late_aircraft', 'pred_total':'total', 'MESSAGE_SHORT':'message' }
    df_print = df_pretty[display_order].rename(columns=rename_map)

    # Present a concise summary table (abbreviated columns)
    summary_cols = ['DATE_OF_FLIGHT', 'SCHEDULED_ARRIVAL_TIME', 'pred_total', 'RISK', 'RISK_LEVEL_SUMMARY', 'MESSAGE']
    summary_cols = [c for c in summary_cols if c in df_output.columns]
    df_summary = df_output[summary_cols].copy()
    # Format date and round total for neat display
    if 'DATE_OF_FLIGHT' in df_summary.columns:
        df_summary['DATE_OF_FLIGHT'] = df_summary['DATE_OF_FLIGHT'].apply(_fmt_date)
    if 'pred_total' in df_summary.columns:
        df_summary['pred_total'] = pd.to_numeric(df_summary['pred_total'], errors='coerce').round(1).fillna(0.0)
    # Shorten MESSAGE for terminal output
    if 'MESSAGE' in df_summary.columns:
        df_summary['MESSAGE'] = df_summary['MESSAGE'].astype(str).apply(lambda s: (s[:100] + '...') if len(s) > 100 else s)

    print(df_summary.head(15).to_string(index=False))

    # Recompute anomalies after adding reconciliation columns
    anomalies = df_output[df_output['RISK']]

    print("\n===== ANOMALY FLIGHTS (RISK=True) =====")

    if anomalies.empty:
        print("No high-risk flights or extreme anomalies detected in the displayed forecast.")
    else:
        print(f"Detected {len(anomalies)} anomalies in the displayed forecast. Showing first 5:")
        # Print concise anomaly lines using the compact MESSAGE token
        for _, r in anomalies.head(5).iterrows():
            try:
                df_date = pd.to_datetime(r.get('DATE_OF_FLIGHT')).strftime('%Y-%m-%d')
            except Exception:
                df_date = str(r.get('DATE_OF_FLIGHT', ''))
            sched = r.get('SCHEDULED_ARRIVAL_TIME', '')
            total = r.get('pred_total', 0.0)
            try:
                total_s = f"{float(total):+.1f}"
            except Exception:
                total_s = str(total)
            recon = r.get('reconciled', False)
            risk = r.get('RISK', False)
            level = r.get('RISK_LEVEL_SUMMARY', '')
            msg = r.get('MESSAGE', '')
            if isinstance(msg, str) and len(msg) > 140:
                msg = msg[:140] + '...'
            print(f"{df_date}  sched:{str(sched):>4}  total:{total_s}  reconciled:{recon}  RISK:{risk}  LEVEL:{level}  MSG: {msg}")
        
    # --- 7. Save Results to CSV Files ---
    out_dir = "forecast_results"
    os.makedirs(out_dir, exist_ok=True)
    
    # Filename components
    safe_airline = selected_airline.replace('/', '_')
    safe_route = selected_route.replace('/', '_')
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    
    # 7a. Save All Predicted Delays (Date | Predicted Delay(s))
    # Determine which prediction columns to persist: prefer predicted-delay columns
    pred_cols = [c for c in df_output.columns if c.endswith('_delay_RATE')] if any(c.endswith('_delay_RATE') for c in df_output.columns) else [c for c in df_output.columns if c.startswith('PRED_') or c=='PREDICTED_DELAY']
    summary_cols = ['DATE_OF_FLIGHT'] + pred_cols
    summary_path = os.path.join(out_dir, f"{safe_airline}_{safe_route}_summaryforfactors_{ts}.csv")
    # SAVE ONLY THE REQUESTED NUMBER OF FLIGHTS
    df_output[summary_cols].to_csv(summary_path, index=False)
    print(f"\n✔ Predicted delays saved to: {summary_path}")

    # 7b. Save Anomalies (Risk)
    risk_cols = ['DATE_OF_FLIGHT', 'SCHEDULED_ARRIVAL_TIME'] + pred_cols + ['MESSAGE', 'RISK']
    risk_path = os.path.join(out_dir, f"{safe_airline}_{safe_route}_risksforfactors_{ts}.csv")
    
    # SAVE ONLY THE ANOMALIES FROM THE REQUESTED FORECAST LENGTH
    anomalies[risk_cols].to_csv(risk_path, index=False)
    print(f"✔ Anomalies/Risk data saved to: {risk_path}")