"""
Streamlit App for End-to-End ML Workflow
Author: Your Name
Description: Modern, professional app for data upload, preprocessing, feature engineering, modeling, and visualization.
"""
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from feature_engineering import *
from preprocessing import *
from utils import *
import optuna
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from lightgbm import log_evaluation

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ML Workflow App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown(
    """
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #4F8BF9; color: white; border-radius: 8px;}
    .stSidebar {background-color: #232946; color: #fff;}
    .stMetric {background: #e0e7ef; border-radius: 8px;}
    .stDataFrame {background: #fff; border-radius: 8px;}
    .stPlotlyChart {background: #fff; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER ---
st.title("🧠 ML Workflow App")
st.markdown("""
A modern, interactive platform for data science workflows: upload data, preprocess, engineer features, train models, and visualize results—all in one place.
""")

# --- SIDEBAR NAVIGATION ---
section = st.sidebar.radio(
    "Navigation",
    ["🔎 Data Exploration", "🧩 Ensemble Models & Visualizations"],
    help="Select a section to work on."
)

# --- SESSION STATE INIT ---
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'features' not in st.session_state:
    st.session_state['features'] = None
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'results' not in st.session_state:
    st.session_state['results'] = None

# --- DATA LOAD & PREPROCESSING ON APP START ---
def load_and_process_data():
    df = pd.read_csv('train.csv')
    st.session_state['data'] = df
    run_preprocessing_and_feature_engineering(df)

# --- AUTOMATIC PREPROCESSING & FEATURE ENGINEERING ---
def run_preprocessing_and_feature_engineering(df):
    # Preprocessing (as in previous logic)
    numerical = df[ [
        'LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
        'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
        'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
        '3SsnPorch','ScreenPorch','PoolArea','MiscVal'] ].copy()
    nominal = df[ [
        'Street','Alley','LandContour','LotConfig','Neighborhood','Condition1',
        'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
        'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',
        'GarageType','GarageFinish','MiscFeature','SaleType','SaleCondition'] ].copy()
    ordinal = df[ [
        'MSSubClass','MSZoning','LotShape','Utilities','LandSlope','OverallQual','OverallCond',
        'YearBuilt','YearRemodAdd','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
        'HeatingQC','Electrical','KitchenQual','Functional','FireplaceQu','GarageYrBlt','GarageFinish','GarageQual','GarageCond',
        'PavedDrive','PoolQC','Fence','MoSold','YrSold'] ].copy()
    numerical_drop = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BedroomAbvGr','GarageCars']
    ordinal_drops = get_low_information_columns(ordinal, 0.7)
    nominal_drops = get_low_information_columns(nominal, 0.7)
    nan_numerical = get_columns_with_excessive_nans(numerical, 0.7)
    nan_nominal = get_columns_with_excessive_nans(nominal, 0.7)
    nan_ordinal = get_columns_with_excessive_nans(ordinal, 0.7)
    tot_num_drop = list(set(numerical_drop + nan_numerical))
    tot_nom_drop = list(set(nominal_drops + nan_nominal))
    tot_ord_drop = list(set(ordinal_drops + nan_ordinal))
    numerical_v2 = numerical.drop(tot_num_drop, axis=1)
    nominal_v2 = nominal.drop(tot_nom_drop, axis=1)
    ordinal_v2 = ordinal.drop(tot_ord_drop, axis=1)
    if 'MoSold' in ordinal_v2.columns:
        ordinal_v2 = ordinal_v2.drop('MoSold', axis=1)
    # Feature engineering
    full_bath = ['BsmtFullBath','FullBath']
    half_bath = ['BsmtHalfBath','HalfBath']
    tot_area = ['LotFrontage','LotArea','MasVnrArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea']
    numerical_v2['total_area'] = numerical_v2[tot_area].sum(axis=1)
    numerical_v2 = numerical_v2.drop(tot_area, axis=1)
    numerical_v2['totalbathreoams'] = numerical_v2[full_bath].sum(axis=1) + (numerical_v2[half_bath].sum(axis=1) * 0.5)
    numerical_v2 = numerical_v2.drop(full_bath, axis=1)
    numerical_v2 = numerical_v2.drop(half_bath, axis=1)
    for col in ['MiscVal', 'GrLivArea']:
        if col in numerical_v2.columns:
            numerical_v2 = numerical_v2.drop(col, axis=1)
    # Nominal encoding
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
    encoded_nominal = pd.DataFrame()
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_data = ohe.fit_transform(nominal_v2)
    encoded_cols = ohe.get_feature_names_out(nominal_v2.columns)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=nominal_v2.index)
    encoded_nominal = pd.concat([encoded_nominal, encoded_df], axis=1)
    # Ordinal encoding (orders_list as provided)
    orders_list = [
        [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190],
        ['Reg','IR1', 'IR2', 'IR3'],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1872, 1875, 1880, 1882, 1885, 1890, 1892, 1893, 1898, 1900, 1904, 1905, 1906, 1908, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917,1918, 1919, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1934, 1935, 1936, 1937, 1938, 1939, 1940,1941, 1942, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964,1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985,1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
        [1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970,1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983,1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
        ['Ex', 'Gd', 'TA', 'Fa'],
        ['Ex', 'Gd', 'TA', 'Fa'],
        ['Gd','Av', 'Mn', 'No'],
        ['GLQ','ALQ', 'BLQ', 'Rec', 'LwQ',  'Unf'],
        ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
        ['Ex', 'Gd', 'TA', 'Fa'],
        ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
        [1900, 1906, 1908, 1910, 1914, 1915, 1916, 1918, 1920, 1921, 1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
        ['Fin', 'RFn', 'Unf'],
        [2006, 2007, 2008, 2009, 2010]
    ]
    if len(orders_list) < len(ordinal_v2.columns):
        orders_list += [None] * (len(ordinal_v2.columns) - len(orders_list))
    encoder = OrdinalEncoder(categories=orders_list, handle_unknown='use_encoded_value', unknown_value=-1)
    ordinal_cols = ordinal_v2.columns
    ordinal_v2[ordinal_cols] = encoder.fit_transform(ordinal_v2[ordinal_cols])
    ordinal_encoded = ordinal_v2
    scaler = StandardScaler()
    numerical_v3 = scaler.fit_transform(numerical_v2)
    numerical_v4 = pd.DataFrame(numerical_v3, columns=numerical_v2.columns)
    final_df = pd.concat([numerical_v4, encoded_nominal, ordinal_encoded], axis=1)
    st.session_state['processed_data'] = {
        'numerical': numerical_v2,
        'nominal': nominal_v2,
        'ordinal': ordinal_v2,
        'tot_num_drop': tot_num_drop,
        'tot_nom_drop': tot_nom_drop,
        'tot_ord_drop': tot_ord_drop
    }
    st.session_state['features'] = final_df
    st.success("Preprocessing, feature engineering, and encoding complete!")
    st.write("Numerical columns after cleaning:")
    st.dataframe(numerical_v2.head(10), use_container_width=True)
    st.write("Nominal columns after cleaning:")
    st.dataframe(nominal_v2.head(10), use_container_width=True)
    st.write("Ordinal columns after cleaning:")
    st.dataframe(ordinal_v2.head(10), use_container_width=True)
    st.write("Final feature matrix (first 20 rows):")
    st.dataframe(final_df.head(20), use_container_width=True)

# --- DATA EXPLORATION SECTION ---
def data_exploration_section():
    st.header("🔎 Data Exploration")
    st.write("The app automatically loads and processes 'train.csv' on startup. Preview your data and basic statistics below.")
    try:
        df = pd.read_csv('train.csv')
        # Only run pipeline if new data or data changed
        if (
            'last_loaded_data_hash' not in st.session_state or
            st.session_state.get('data') is None or
            hash(df.to_csv(index=False)) != st.session_state.get('last_loaded_data_hash')
        ):
            st.session_state['data'] = df
            st.session_state['last_loaded_data_hash'] = hash(df.to_csv(index=False))
            run_preprocessing_and_feature_engineering(df)
        st.success("'train.csv' loaded and processed successfully!")
        with st.expander("Preview Data"):
            st.dataframe(st.session_state['data'].style.set_properties(**{'background-color': '#f8f9fa'}), use_container_width=True)
        with st.expander("Basic Statistics"):
            st.dataframe(st.session_state['data'].describe().T, use_container_width=True)
        st.metric("Rows", st.session_state['data'].shape[0])
        st.metric("Columns", st.session_state['data'].shape[1])
        # --- Correlation Matrix Heatmap ---
        import plotly.express as px

        processed_data = st.session_state.get('processed_data')
        if processed_data is not None and 'numerical' in processed_data:
            numerical_v2 = processed_data['numerical']
            if not numerical_v2.empty:
                corr_matrix = numerical_v2.corr(numeric_only=True)
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix Heatmap',
                    aspect='auto'
                )
                fig.update_layout(
                    xaxis_title='Features',
                    yaxis_title='Features',
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numerical data available for correlation matrix.")
        else:
            st.info("Processed data not available. Please reload the page if this persists.")
    except Exception as e:
        st.error(f"Error loading 'train.csv': {e}")

# --- ENSEMBLE MODELS & VISUALIZATIONS SECTION ---
def ensemble_models_visualizations_section():
    st.header("🧩 Ensemble Models & Visualizations")
    st.subheader("Average Majority Voting Ensemble")
    st.write("This ensemble combines SVR, Linear Regression, and Random Forest Regressor (random_state=42) using simple averaging of their predictions.")

    from sklearn.svm import SVR
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    import numpy as np

    # Use processed features and target
    features = st.session_state.get('features')
    data = st.session_state.get('data')
    if features is not None and data is not None and 'SalePrice' in data.columns:
        X = features
        y = data['SalePrice']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        svr = SVR()
        lr = LinearRegression()
        rf = RandomForestRegressor(random_state=42)

        # Fit models
        svr.fit(X_train, y_train)
        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)

        # Predict
        pred_svr = svr.predict(X_test)
        pred_lr = lr.predict(X_test)
        pred_rf = rf.predict(X_test)
        # Average predictions
        avg_pred = (pred_svr + pred_lr + pred_rf) / 3

        # Metrics
        mse = mean_squared_error(y_test, avg_pred)
        r2 = r2_score(y_test, avg_pred)
        # --- Custom styled metrics for better visibility (Averaged Ensemble) ---
        st.markdown("""
        <div style='display: flex; gap: 2em;'>
            <div style='background: #1f77b4; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
                <div style='font-size: 1.2em;'>Test MSE (Averaged Ensemble)</div>
                <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
            </div>
            <div style='background: #2ca02c; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
                <div style='font-size: 1.2em;'>Test RMSE (Averaged Ensemble)</div>
                <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
            </div>
            <div style='background: #d62728; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
                <div style='font-size: 1.2em;'>Test R² (Averaged Ensemble)</div>
                <div style='font-size: 2em; font-weight: bold;'>{:.3f}</div>
            </div>
        </div>
        """.format(mse, np.sqrt(mse), r2), unsafe_allow_html=True)

        # --- PLOTS ---
        import plotly.graph_objects as go
        # 1. Actual vs Predicted Scatter
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=y_test, y=avg_pred,
                                  mode='markers',
                                  name='Predicted vs Actual',
                                  marker=dict(color='blue', opacity=0.6)))
        fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                  y=[y_test.min(), y_test.max()],
                                  mode='lines',
                                  name='Perfect Prediction',
                                  line=dict(color='red', dash='dash')))
        fig1.update_layout(title="Actual vs Predicted (Averaged Ensemble)",
                          xaxis_title="Actual",
                          yaxis_title="Predicted",
                          template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Residual Plot
        residuals = y_test - avg_pred
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=avg_pred, y=residuals,
                                  mode='markers',
                                  name='Residuals',
                                  marker=dict(color='orange', opacity=0.6)))
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(title="Residual Plot",
                          xaxis_title="Predicted",
                          yaxis_title="Residuals",
                          template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Model MSE Comparison Bar Chart
        models = ['Linear Regression', 'Random Forest', 'SVR', 'Ensemble Average']
        mses = [
            mean_squared_error(y_test, pred_lr),
            mean_squared_error(y_test, pred_rf),
            mean_squared_error(y_test, pred_svr),
            mean_squared_error(y_test, avg_pred)
        ]
        fig3 = go.Figure([go.Bar(x=models, y=mses, marker_color=['skyblue', 'lightgreen', 'salmon', 'purple'])])
        fig3.update_layout(title="Model MSE Comparison",
                          xaxis_title="Model",
                          yaxis_title="MSE",
                          template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Features or target not available. Please ensure data is loaded and processed.")

    st.subheader("Average Majority Voting Ensemble (XGB, LGB, LR, RF)")
    st.write("This ensemble combines XGBoost, LightGBM, Linear Regression, and Random Forest Regressor (random_state=42) using simple averaging of their predictions.")

    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    if features is not None and data is not None and 'SalePrice' in data.columns:
        X = features
        y = data['SalePrice']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define models
        xgb = XGBRegressor(random_state=42, verbosity=0)
        lgb = LGBMRegressor(random_state=42)
        lr2 = LinearRegression()
        rf2 = RandomForestRegressor(random_state=42)

        # Fit models
        xgb.fit(X_train, y_train)
        lgb.fit(X_train, y_train)
        lr2.fit(X_train, y_train)
        rf2.fit(X_train, y_train)

        # Predict
        pred_xgb = xgb.predict(X_test)
        pred_lgb = lgb.predict(X_test)
        pred_lr2 = lr2.predict(X_test)
        pred_rf2 = rf2.predict(X_test)
        # Average predictions
        avg_pred2 = (pred_xgb + pred_lgb + pred_lr2 + pred_rf2) / 4

        # Metrics
        mse2 = mean_squared_error(y_test, avg_pred2)
        r2_2 = r2_score(y_test, avg_pred2)
        # --- Custom styled metrics for better visibility (Averaged Ensemble XGB/LGB/LR/RF) ---
        st.markdown("""
        <div style='display: flex; gap: 2em;'>
            <div style='background: #1f77b4; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
                <div style='font-size: 1.2em;'>Test MSE (Averaged Ensemble XGB/LGB/LR/RF)</div>
                <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
            </div>
            <div style='background: #2ca02c; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
                <div style='font-size: 1.2em;'>Test RMSE (Averaged Ensemble XGB/LGB/LR/RF)</div>
                <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
            </div>
            <div style='background: #d62728; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
                <div style='font-size: 1.2em;'>Test R² (Averaged Ensemble XGB/LGB/LR/RF)</div>
                <div style='font-size: 2em; font-weight: bold;'>{:.3f}</div>
            </div>
        </div>
        """.format(mse2, np.sqrt(mse2), r2_2), unsafe_allow_html=True)

        # --- PLOTS ---
        # 1. Actual vs Predicted Scatter
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=y_test, y=avg_pred2,
                                  mode='markers',
                                  name='Predicted vs Actual',
                                  marker=dict(color='blue', opacity=0.6)))
        fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                  y=[y_test.min(), y_test.max()],
                                  mode='lines',
                                  name='Perfect Prediction',
                                  line=dict(color='red', dash='dash')))
        fig1.update_layout(title="Actual vs Predicted (Averaged Ensemble XGB/LGB/LR/RF)",
                          xaxis_title="Actual",
                          yaxis_title="Predicted",
                          template="plotly_white")
        st.plotly_chart(fig1, use_container_width=True)

        # 2. Residual Plot
        residuals2 = y_test - avg_pred2
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=avg_pred2, y=residuals2,
                                  mode='markers',
                                  name='Residuals',
                                  marker=dict(color='orange', opacity=0.6)))
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        fig2.update_layout(title="Residual Plot (Averaged Ensemble XGB/LGB/LR/RF)",
                          xaxis_title="Predicted",
                          yaxis_title="Residuals",
                          template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

        # 3. Model MSE Comparison Bar Chart
        models2 = ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Ensemble Average']
        mses2 = [
            mean_squared_error(y_test, pred_lr2),
            mean_squared_error(y_test, pred_rf2),
            mean_squared_error(y_test, pred_xgb),
            mean_squared_error(y_test, pred_lgb),
            mean_squared_error(y_test, avg_pred2)
        ]
        fig3 = go.Figure([go.Bar(x=models2, y=mses2, marker_color=['skyblue', 'lightgreen', 'goldenrod', 'darkgreen', 'purple'])])
        fig3.update_layout(title="Model MSE Comparison (XGB/LGB/LR/RF)",
                          xaxis_title="Model",
                          yaxis_title="MSE",
                          template="plotly_white")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("Features or target not available. Please ensure data is loaded and processed.")

    # --- WEIGHTED ENSEMBLE (OPTIMIZED WEIGHTS) ---
    st.subheader("Weighted Averaging Ensemble (Optimized Weights: LR, RF, XGB, LGB)")
    st.write("This ensemble finds the optimal weights for combining Linear Regression, Random Forest, XGBoost, and LightGBM predictions using scipy.optimize.minimize to maximize R².")

    # Prepare predictions (already computed above)
    lr_pred = pred_lr2
    rf_pred = pred_rf2
    xgb_pred = pred_xgb
    lgb_pred = pred_lgb

    def objective(weights):
        weighted_pred = (
            weights[0] * lr_pred +
            weights[1] * rf_pred +
            weights[2] * xgb_pred +
            weights[3] * lgb_pred
        )
        return -r2_score(y_test, weighted_pred)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0,1)] * 4
    init_guess = [0.25, 0.25, 0.25, 0.25]
    result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)
    best_weights = result.x
    best_r2 = -result.fun
    final_pred = (
        best_weights[0] * lr_pred +
        best_weights[1] * rf_pred +
        best_weights[2] * xgb_pred +
        best_weights[3] * lgb_pred
    )
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, final_pred)
    rmse = np.sqrt(mse)

    # --- Custom styled metrics for better visibility (Weighted Ensemble) ---
    st.markdown("""
    <div style='display: flex; gap: 2em;'>
        <div style='background: #1f77b4; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Weighted Ensemble MSE</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
        </div>
        <div style='background: #2ca02c; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Weighted Ensemble RMSE</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
        </div>
        <div style='background: #d62728; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Optimized R² (Weighted Ensemble)</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.3f}</div>
        </div>
    </div>
    """.format(mse, rmse, best_r2), unsafe_allow_html=True)

    # --- PLOTS FOR WEIGHTED ENSEMBLE ---
    import plotly.express as px
    import plotly.graph_objects as go
    model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM']
    # 1. Optimized Model Weights Bar Chart
    fig_weights = go.Figure([go.Bar(x=model_names, y=best_weights,
                                    marker_color=['skyblue', 'lightgreen', 'goldenrod', 'forestgreen'])])
    fig_weights.update_layout(title="Optimized Model Weights (Weighted Ensemble)",
                      xaxis_title="Model",
                      yaxis_title="Weight",
                      template="plotly_white")
    st.plotly_chart(fig_weights, use_container_width=True)

    # 2. Residual Distribution Histogram (with boxplot)
    residuals = y_test - final_pred
    fig_resid = px.histogram(residuals, nbins=30, marginal='box',
                       title="Residual Distribution (Weighted Ensemble)",
                       labels={'value': 'Prediction Error'},
                       template="plotly_white")
    fig_resid.update_layout(xaxis_title="Residual", yaxis_title="Frequency")
    st.plotly_chart(fig_resid, use_container_width=True)

    # 3. Simple vs Weighted Ensemble Performance (MSE and R², dual y-axis bar chart)
    from sklearn.metrics import r2_score, mean_squared_error
    compare_df = {
        'Model': ['Simple Avg', 'Weighted Avg'],
        'MSE': [mean_squared_error(y_test, avg_pred2), mean_squared_error(y_test, final_pred)],
        'R2': [r2_score(y_test, avg_pred2), r2_score(y_test, final_pred)]
    }
    import pandas as pd
    df_compare = pd.DataFrame(compare_df)
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Bar(x=df_compare['Model'], y=df_compare['MSE'], name='MSE', marker_color='lightsalmon'))
    fig_compare.add_trace(go.Bar(x=df_compare['Model'], y=df_compare['R2'], name='R² Score', marker_color='lightblue', yaxis='y2'))
    fig_compare.update_layout(
        title="Simple vs Weighted Ensemble Performance",
        xaxis_title="Model",
        yaxis=dict(title='MSE', side='left'),
        yaxis2=dict(title='R² Score', overlaying='y', side='right'),
        barmode='group',
        template='plotly_white'
    )
    st.plotly_chart(fig_compare, use_container_width=True)

    # 4. XGBoost Learning Curve (RMSE) - fix: pass eval_metric to constructor, not fit()
    xgb_lc = XGBRegressor(random_state=42, verbosity=0, eval_metric='rmse')
    xgb_lc.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    xgb_eval = xgb_lc.evals_result()
    fig_xgb_lc = go.Figure()
    fig_xgb_lc.add_trace(go.Scatter(y=xgb_eval['validation_0']['rmse'], name='XGB Train RMSE'))
    fig_xgb_lc.add_trace(go.Scatter(y=xgb_eval['validation_1']['rmse'], name='XGB Test RMSE'))
    fig_xgb_lc.update_layout(title="XGBoost Learning Curve (RMSE)",
                      xaxis_title="Boosting Round",
                      yaxis_title="RMSE",
                      template="plotly_white")
    st.plotly_chart(fig_xgb_lc, use_container_width=True)

    # 5. LightGBM Learning Curve (RMSE) - using evals_result_ as suggested
    lgb_lc = LGBMRegressor(random_state=42)
    lgb_lc.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='rmse', callbacks=[log_evaluation(period=0)])
    lgb_eval = lgb_lc.evals_result_
    fig_lgb_lc = go.Figure()
    fig_lgb_lc.add_trace(go.Scatter(y=lgb_eval['training']['rmse'], name='LGBM Train RMSE'))
    fig_lgb_lc.add_trace(go.Scatter(y=lgb_eval['valid_1']['rmse'], name='LGBM Test RMSE'))
    fig_lgb_lc.update_layout(title="LightGBM Learning Curve (RMSE)",
                      xaxis_title="Boosting Round",
                      yaxis_title="RMSE",
                      template="plotly_white")
    st.plotly_chart(fig_lgb_lc, use_container_width=True)
    # End weighted ensemble plots

    # --- VOTING REGRESSOR ENSEMBLE ---
    st.subheader("Voting Regressor Ensemble (LR, RF, XGB, LGB)")
    st.write("This ensemble combines Linear Regression, Random Forest, XGBoost, and LightGBM using scikit-learn's VotingRegressor.")

    from sklearn.ensemble import VotingRegressor
    # Use the same train/test split and features as above
    voting_reg = VotingRegressor(estimators=[
        ('lr', lr2),
        ('rf', rf2),
        ('xgb', xgb),
        ('lgb', lgb)
    ])
    voting_reg.fit(X_train, y_train)
    voting_pred = voting_reg.predict(X_test)

    voting_mse = mean_squared_error(y_test, voting_pred)
    voting_rmse = np.sqrt(voting_mse)
    voting_r2 = r2_score(y_test, voting_pred)

    # --- Custom styled metrics for better visibility (Voting Regressor) ---
    st.markdown("""
    <div style='display: flex; gap: 2em;'>
        <div style='background: #1f77b4; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Voting Regressor MSE</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
        </div>
        <div style='background: #2ca02c; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Voting Regressor RMSE</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
        </div>
        <div style='background: #d62728; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Voting Regressor R² Score</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.4f}</div>
        </div>
    </div>
    """.format(voting_mse, voting_rmse, voting_r2), unsafe_allow_html=True)

    # --- Voting Regressor Plot: True vs Predicted ---
    fig_voting = go.Figure()
    fig_voting.add_trace(go.Scatter(y=y_test.values, mode='lines+markers', name='True Values'))
    fig_voting.add_trace(go.Scatter(y=voting_pred, mode='lines+markers', name='Voting Regressor'))
    fig_voting.update_layout(title="True vs Voting Regressor Predictions",
                            xaxis_title="Test Sample Index",
                            yaxis_title="Target Value",
                            template="plotly_white")
    st.plotly_chart(fig_voting, use_container_width=True)
    # End Voting Regressor section

    # --- STACKING REGRESSOR ENSEMBLE ---
    st.subheader("Stacking Regressor Ensemble (RF, XGB, LGB + Ridge Meta-Model)")
    st.write("This ensemble stacks Random Forest, XGBoost, and LightGBM as base models with Ridge Regression as the meta-model (cv=10).")

    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import Ridge
    # Define base models and meta-model
    stacking_estimators = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('xgb', XGBRegressor(random_state=42)),
        ('lgb', LGBMRegressor(random_state=42))
    ]
    meta_regressor = Ridge(alpha=1.0)
    stacking_reg = StackingRegressor(
        estimators=stacking_estimators,
        final_estimator=meta_regressor,
        cv=10,
        n_jobs=-1
    )
    stacking_reg.fit(X_train, y_train)
    stacking_pred = stacking_reg.predict(X_test)

    stacking_mse = mean_squared_error(y_test, stacking_pred)
    stacking_rmse = np.sqrt(stacking_mse)
    stacking_r2 = r2_score(y_test, stacking_pred)

    # --- Custom styled metrics for better visibility (Stacking Regressor) ---
    st.markdown("""
    <div style='display: flex; gap: 2em;'>
        <div style='background: #1f77b4; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Stacking Regressor MSE</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
        </div>
        <div style='background: #2ca02c; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Stacking Regressor RMSE</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
        </div>
        <div style='background: #d62728; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Stacking Regressor R² Score</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.4f}</div>
        </div>
    </div>
    """.format(stacking_mse, stacking_rmse, stacking_r2), unsafe_allow_html=True)

    # --- Residuals Distribution Plot ---
    stacking_residuals = y_test - stacking_pred
    fig_stack_resid = go.Figure()
    fig_stack_resid.add_trace(go.Histogram(x=stacking_residuals, nbinsx=30, name='Residuals'))
    fig_stack_resid.update_layout(title="Residuals Distribution of Stacking Regressor",
                                 xaxis_title="Residual",
                                 yaxis_title="Frequency",
                                 template="plotly_white")
    st.plotly_chart(fig_stack_resid, use_container_width=True)

    # --- Feature Importances from Base Models ---
    feature_names = list(X_train.columns)
    rf = RandomForestRegressor(random_state=42)
    xgb = XGBRegressor(random_state=42, verbosity=0)
    lgb = LGBMRegressor(random_state=42)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    lgb.fit(X_train, y_train)
    rf_importances = rf.feature_importances_
    xgb_importances = xgb.feature_importances_
    lgb_importances = lgb.feature_importances_
    fig_stack_feat = go.Figure()
    fig_stack_feat.add_trace(go.Bar(x=feature_names, y=rf_importances, name="Random Forest"))
    fig_stack_feat.add_trace(go.Bar(x=feature_names, y=xgb_importances, name="XGBoost"))
    fig_stack_feat.add_trace(go.Bar(x=feature_names, y=lgb_importances, name="LightGBM"))
    fig_stack_feat.update_layout(
        title="Feature Importances from Base Models",
        xaxis_title="Features",
        yaxis_title="Importance",
        barmode='group',
        template="plotly_white",
        xaxis_tickangle=-45,
        height=500
    )
    st.plotly_chart(fig_stack_feat, use_container_width=True)
    # End stacking regressor section

    # --- BLENDING ENSEMBLE (RF, XGB, LGB + Ridge Meta-Model) ---
    st.subheader("Blending Ensemble (RF, XGB, LGB + Ridge Meta-Model)")
    st.write("This ensemble uses a blending approach: base models are trained on a subset of the training data, and a meta-model (Ridge) is trained on their predictions on a holdout set.")

    # Step 1: Split training data into base training and holdout (for blending)
    X_base_train, X_holdout, y_base_train, y_holdout = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Step 2: Initialize base models
    rf_blend = RandomForestRegressor(random_state=42)
    xgb_blend = XGBRegressor(random_state=42)
    lgb_blend = LGBMRegressor(random_state=42)

    # Step 3: Train base models on the base training set only
    rf_blend.fit(X_base_train, y_base_train)
    xgb_blend.fit(X_base_train, y_base_train)
    lgb_blend.fit(X_base_train, y_base_train)

    # Step 4: Generate predictions of base models on the holdout set (meta-model training data)
    rf_holdout_pred = rf_blend.predict(X_holdout)
    xgb_holdout_pred = xgb_blend.predict(X_holdout)
    lgb_holdout_pred = lgb_blend.predict(X_holdout)

    # Step 5: Stack these predictions as features to train the meta-model (blender)
    X_blend_train = np.column_stack((rf_holdout_pred, xgb_holdout_pred, lgb_holdout_pred))

    # Step 6: Initialize and train meta-model (e.g., Ridge)
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_blend_train, y_holdout)

    # Step 7: Predict base models on the original test set
    rf_test_pred = rf_blend.predict(X_test)
    xgb_test_pred = xgb_blend.predict(X_test)
    lgb_test_pred = lgb_blend.predict(X_test)

    # Step 8: Stack test predictions as input for meta-model
    X_blend_test = np.column_stack((rf_test_pred, xgb_test_pred, lgb_test_pred))

    # Step 9: Make final blended predictions
    y_blend_pred = meta_model.predict(X_blend_test)

    # Step 10: Evaluate the blended predictions
    blend_mse = mean_squared_error(y_test, y_blend_pred)
    blend_rmse = np.sqrt(blend_mse)
    blend_r2 = r2_score(y_test, y_blend_pred)

    # --- Custom styled metrics for better visibility ---
    st.markdown("""
    <div style='display: flex; gap: 2em;'>
        <div style='background: #1f77b4; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Blending MSE</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
        </div>
        <div style='background: #2ca02c; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Blending RMSE</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.2f}</div>
        </div>
        <div style='background: #d62728; color: white; padding: 1em 2em; border-radius: 10px; text-align: center;'>
            <div style='font-size: 1.2em;'>Blending R² Score</div>
            <div style='font-size: 2em; font-weight: bold;'>{:.4f}</div>
        </div>
    </div>
    """.format(blend_mse, blend_rmse, blend_r2), unsafe_allow_html=True)

    # --- PLOTS FOR BLENDING ENSEMBLE ---
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd
    # 1. Correlation Matrix of Base Model Predictions
    preds_df = pd.DataFrame({
        'RandomForest': rf_test_pred,
        'XGBoost': xgb_test_pred,
        'LightGBM': lgb_test_pred
    })
    fig_corr = px.imshow(preds_df.corr(), 
                        text_auto=True, 
                        color_continuous_scale='Viridis', 
                        title='Correlation Matrix of Base Model Predictions')
    st.plotly_chart(fig_corr, use_container_width=True)

    # 2. Meta-Model Coefficients (Blender Weights)
    coef_df = pd.DataFrame({
        'Model': ['RandomForest', 'XGBoost', 'LightGBM'],
        'Coefficient': meta_model.coef_
    })
    fig_coef = px.bar(coef_df, x='Model', y='Coefficient',
                     title='Meta-Model Coefficients (Blender Weights)',
                     text='Coefficient')
    st.plotly_chart(fig_coef, use_container_width=True)

    # 3. Prediction Errors Distribution by Model
    errors_df = pd.DataFrame({
        'RandomForest Error': np.abs(y_test - rf_test_pred),
        'XGBoost Error': np.abs(y_test - xgb_test_pred),
        'LightGBM Error': np.abs(y_test - lgb_test_pred),
        'Blender Error': np.abs(y_test - y_blend_pred)
    })
    fig_errors = go.Figure()
    for col in errors_df.columns:
        fig_errors.add_trace(go.Box(y=errors_df[col], name=col))
    fig_errors.update_layout(title='Prediction Errors Distribution by Model',
                            yaxis_title='Absolute Error',
                            template='plotly_white')
    st.plotly_chart(fig_errors, use_container_width=True)

    # 4. Error Improvement by Blending over Best Base Model
    best_base_error = np.minimum.reduce([
        np.abs(y_test - rf_test_pred),
        np.abs(y_test - xgb_test_pred),
        np.abs(y_test - lgb_test_pred)
    ])
    improvement = best_base_error - np.abs(y_test - y_blend_pred)  # Positive means blender better
    fig_improve = px.histogram(improvement, nbins=30, 
                       title='Error Improvement by Blending over Best Base Model',
                       labels={'value':'Improvement in Absolute Error'})
    st.plotly_chart(fig_improve, use_container_width=True)

    # 5. Scatter of RF vs XGB Predictions
    fig_scatter = px.scatter(x=rf_test_pred, y=xgb_test_pred,
                     labels={'x':'RandomForest Predictions', 'y':'XGBoost Predictions'},
                     title='Scatter of RF vs XGB Predictions')
    st.plotly_chart(fig_scatter, use_container_width=True)
# --- MAIN APP ROUTER ---
if 'data' not in st.session_state or st.session_state['data'] is None:
    load_and_process_data()
if section == "🔎 Data Exploration":
    data_exploration_section()
elif section == "🧩 Ensemble Models & Visualizations":
    ensemble_models_visualizations_section()