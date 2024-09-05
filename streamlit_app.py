import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import altair as alt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# ปิดข้อความเตือนทั้งหมด
warnings.filterwarnings("ignore")

def load_data(file):
    return pd.read_csv(file)

def check_missing_values(df, step="Initial"):
    """Check and print missing values in the DataFrame."""
    missing_values = df.isnull().sum()
    st.write(f"Missing values at {step}:")
    st.write(missing_values)

def clean_data(df):
    data_clean = df.copy()
    data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')
    data_clean = data_clean.dropna(subset=['datetime'])
    data_clean = data_clean[(data_clean['wl_up'] >= 100) & (data_clean['wl_up'] <= 450)]
    data_clean = data_clean[(data_clean['wl_up'] != 0) & (~data_clean['wl_up'].isna())]
    return data_clean

def create_time_features(data_clean):
    if not pd.api.types.is_datetime64_any_dtype(data_clean['datetime']):
        data_clean['datetime'] = pd.to_datetime(data_clean['datetime'], errors='coerce')

    data_clean['year'] = data_clean['datetime'].dt.year
    data_clean['month'] = data_clean['datetime'].dt.month
    data_clean['day'] = data_clean['datetime'].dt.day
    data_clean['hour'] = data_clean['datetime'].dt.hour
    data_clean['minute'] = data_clean['datetime'].dt.minute
    data_clean['day_of_week'] = data_clean['datetime'].dt.dayofweek
    data_clean['day_of_year'] = data_clean['datetime'].dt.dayofyear
    data_clean['week_of_year'] = data_clean['datetime'].dt.isocalendar().week
    data_clean['days_in_month'] = data_clean['datetime'].dt.days_in_month

    # st.write("Missing values after creating time features:")
    # st.write(data_clean.isnull().sum())

    return data_clean

def prepare_features(data_clean):
    feature_cols = [
        'year', 'month', 'day', 'hour', 'minute',
        'day_of_week', 'day_of_year', 'week_of_year',
        'days_in_month'
    ]
    X = data_clean[feature_cols]
    y = data_clean['wl_up']
    return X, y

def train_model(X_train, y_train):
    """Train model with RandomizedSearchCV for hyperparameter tuning."""
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=42)

    n_splits = min(3, len(X_train) // 2)  # Ensuring at least 2 folds if possible
    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=10, cv=n_splits, n_jobs=-1, verbose=2, random_state=42)
    random_search.fit(X_train, y_train)

    # st.write("Best parameters found: ", random_search.best_params_)
    # st.write("Best score found: ", random_search.best_score_)

    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error (MSE): {mse:.4f}")
    st.write(f"Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"R-squared (R²): {r2:.4f}")

def generate_missing_dates(data):
    full_date_range = pd.date_range(start=data['datetime'].min(), end=data['datetime'].max(), freq='15T')
    all_dates = pd.DataFrame(full_date_range, columns=['datetime'])
    data_with_all_dates = pd.merge(all_dates, data, on='datetime', how='left')
    return data_with_all_dates

def fill_code_column(data):
    data['code'] = data['code'].fillna(method='ffill').fillna(method='bfill')
    return data

def apply_ema_and_sma(data, ema_span=12, sma_window=12):
    data['wl_up'] = data['wl_up'].ewm(span=ema_span, adjust=False).mean()
    data['wl_up'] = data['wl_up'].rolling(window=sma_window, min_periods=1).mean()
    return data

def apply_median_filter(data, window_size=5):
    data['wl_up'] = data['wl_up'].rolling(window=window_size, min_periods=1, center=True).median()
    return data

# def fill_outliers_with_rf_and_smooth(data_clean):
#     feature_cols = ['year', 'month', 'day', 'hour', 'minute',
#                     'day_of_week', 'day_of_year', 'week_of_year', 'days_in_month']

#     Q1 = data_clean['wl_up'].quantile(0.25)
#     Q3 = data_clean['wl_up'].quantile(0.75)
#     IQR = Q3 - Q1
#     outliers = ((data_clean['wl_up'] < (Q1 - 1.5 * IQR)) | (data_clean['wl_up'] > (Q3 + 1.5 * IQR)))

#     data_clean.loc[outliers, 'wl_up'] = np.nan
#     return data_clean

def handle_missing_values(data_clean, start_date, end_date):
    feature_cols = ['year', 'month', 'day', 'hour', 'minute',
                    'day_of_week', 'day_of_year', 'week_of_year', 'days_in_month']

    # ลบ timezone ออกจากข้อมูล datetime
    data_clean['datetime'] = data_clean['datetime'].dt.tz_localize(None)

    # แปลง start_date และ end_date ให้เป็น timezone-aware
    start_date = pd.Timestamp(start_date).tz_localize(None)
    end_date = pd.Timestamp(end_date).tz_localize(None)

    data = data_clean[(data_clean['datetime'] >= start_date) & (data_clean['datetime'] <= end_date)]

    # Generate all dates and merge with existing data
    data_with_all_dates = generate_missing_dates(data)

    # Convert index to datetime index
    data_with_all_dates.index = pd.to_datetime(data_with_all_dates.index)

    # Separate missing and non-missing data
    data_missing = data_with_all_dates[data_with_all_dates['wl_up'].isnull()]
    data_not_missing = data_with_all_dates.dropna(subset=['wl_up'])

    if len(data_missing) == 0:
        print("No missing values to predict.")
        return data_with_all_dates

    # Fill missing values iteratively by day
    missing_dates = sorted(data_missing['datetime'].dt.date.unique())
    for date in missing_dates:
        print(f"Handling missing values for {date}")

        # Get the start and end dates for training
        start_train_date = date - pd.DateOffset(days=7)
        end_train_date = date - pd.DateOffset(days=1)
        data_train = data_not_missing[(data_not_missing['datetime'].dt.date >= start_train_date.date()) & 
                                      (data_not_missing['datetime'].dt.date <= end_train_date.date())]

        if len(data_train) == 0:
            # If not enough historical data, use whatever is available
            data_train = data_not_missing

        # Prepare features and targets for non-missing data
        X_train, y_train = prepare_features(data_train)
        X_train_scaled = StandardScaler().fit_transform(X_train)
        model = train_model(X_train_scaled, y_train)

        # Prepare features for the missing day
        day_start = pd.Timestamp(date).tz_localize(None)
        day_end = day_start + pd.DateOffset(days=1) - pd.Timedelta(minutes=1)
        X_missing = data_missing[(data_missing['datetime'] >= day_start) & (data_missing['datetime'] <= day_end)][feature_cols]
        X_missing_scaled = StandardScaler().fit_transform(X_missing)

        if X_missing_scaled.shape[0] == len(data_missing[(data_missing['datetime'] >= day_start) & (data_missing['datetime'] <= day_end)]):
            data_with_all_dates.loc[data_missing[(data_missing['datetime'] >= day_start) & (data_missing['datetime'] <= day_end)].index, 'wl_up'] = model.predict(X_missing_scaled)

    # Apply smoothing techniques
    data_with_all_dates = apply_ema_and_sma(data_with_all_dates, ema_span=20, sma_window=20)
    data_with_all_dates = apply_median_filter(data_with_all_dates, window_size=5)

    # Reset index to integer
    data_with_all_dates.reset_index(drop=True, inplace=True)
    return data_with_all_dates

def randomly_delete_data(data, start_date, end_date):
    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # กรองข้อมูลเฉพาะช่วงเวลาที่ระบุ
    data_filtered = data[(data['datetime'] >= start_date) & (data['datetime'] <= end_date)]

    # เรียงข้อมูลตามสัปดาห์
    weekly_groups = data_filtered.groupby('week_of_year')

    # ลบข้อมูลในแต่ละสัปดาห์
    for week, group in weekly_groups:
        # จำนวนวันที่ต้องการลบในแต่ละสัปดาห์ (สุ่มระหว่าง 0-3 วัน)
        min_days = 0
        max_days = 3
        num_days_to_delete = np.random.randint(min_days, max_days + 1)

        # จำนวนแถวในกลุ่ม (สัปดาห์) ที่จะลบ
        num_rows = len(group)
        num_rows_to_delete = num_days_to_delete * 96

        # ตรวจสอบว่าจำนวนแถวเพียงพอหรือไม่
        if num_rows >= num_rows_to_delete and num_rows_to_delete > 0:
            # สุ่มตำแหน่งเริ่มต้น
            random_start_idx = np.random.randint(0, num_rows - num_rows_to_delete + 1)
            delete_indices = group.index[random_start_idx:random_start_idx + num_rows_to_delete]

            # ลบข้อมูล
            data.loc[delete_indices, 'wl_up'] = np.nan

    return data

def plot_results(data_before, data_filled, data_deleted):
    data_before_filled = pd.DataFrame({
        'วันที่': data_before['datetime'],
        'ข้อมูลเดิม': data_before['wl_up']
    })

    data_after_filled = pd.DataFrame({
        'วันที่': data_filled['datetime'],
        'ข้อมูลหลังเติมค่า': data_filled['wl_up']
    })

    data_after_deleted = pd.DataFrame({
        'วันที่': data_deleted['datetime'],
        'ข้อมูลหลังสุ่มลบ': data_deleted['wl_up']
    })

    combined_data = pd.merge(data_before_filled, data_after_filled, on='วันที่', how='outer')
    combined_data = pd.merge(combined_data, data_after_deleted, on='วันที่', how='outer')

    min_y = combined_data[['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า', 'ข้อมูลหลังสุ่มลบ']].min().min()
    max_y = combined_data[['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า', 'ข้อมูลหลังสุ่มลบ']].max().max()

    chart = alt.Chart(combined_data).transform_fold(
        ['ข้อมูลเดิม', 'ข้อมูลหลังเติมค่า', 'ข้อมูลหลังสุ่มลบ'],
        as_=['ข้อมูล', 'ระดับน้ำ']
    ).mark_line().encode(
        x='วันที่:T',
        y=alt.Y('ระดับน้ำ:Q', scale=alt.Scale(domain=[min_y, max_y])),
        color=alt.Color('ข้อมูล:N',legend=alt.Legend(orient='bottom', title='ข้อมูล'))
    ).properties(
        title='ข้อมูลหลังจากการเติมค่าที่หายไป',
        height=400
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    st.write("ตารางแสดงข้อมูลหลังเติมค่า")
    st.dataframe(data_filled)

def plot_data_preview(df):
    min_y = df['wl_up'].min()
    max_y = df['wl_up'].max()

    chart = alt.Chart(df).mark_line(color='#ffabab').encode(
        x=alt.X('datetime:T', title='วันที่'),
        y=alt.Y('wl_up:Q', scale=alt.Scale(domain=[min_y, max_y]), title='ระดับน้ำ')
    ).properties(
        title='ตัวอย่างข้อมูล'
    )

    st.altair_chart(chart, use_container_width=True)

# Streamlit UI
st.title("การจัดการกับข้อมูลระดับน้ำด้วย Random Forest (week)")

uploaded_file = st.file_uploader("เลือกไฟล์ CSV", type="csv")

if uploaded_file is not None:
    df = load_data(uploaded_file)
    df_pre = clean_data(df)
    df_pre = generate_missing_dates(df_pre)
    df_pre = fill_code_column(df_pre)
    df_pre = create_time_features(df_pre)
    plot_data_preview(df_pre)

    start_date = st.date_input("วันที่เริ่มต้น", value=pd.to_datetime("2023-10-01"))
    end_date = st.date_input("วันที่สิ้นสุด", value=pd.to_datetime("2023-11-30"))

    if st.button("เลือก"):
        st.markdown("---")
        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
        
        df_filtered = df[(df['datetime'] >= pd.to_datetime(start_date)) & (df['datetime'] <= pd.to_datetime(end_date))]

        # st.write("Filtered Data Preview:")
        # st.write(df_filtered.head())

        # Check missing values initially
        # check_missing_values(df_filtered)

        # Clean data
        df_clean = clean_data(df_filtered)

        # Generate all dates
        df_clean = generate_missing_dates(df_clean)

        # Fill NaN values in 'code' column
        df_clean = fill_code_column(df_clean)

        # Create time features
        df_clean = create_time_features(df_clean)

        # เก็บข้อมูลก่อนการสุ่มลบ
        df_before_random_deletion = df_filtered.copy()

        # Randomly delete data
        df_deleted = randomly_delete_data(df_clean, start_date, end_date)
        
        # Handle missing values by week
        df_handled = handle_missing_values(df_clean, start_date, end_date)

        # Plot the results using Streamlit's line chart
        plot_results(df_before_random_deletion, df_handled, df_deleted)
    st.markdown("---")