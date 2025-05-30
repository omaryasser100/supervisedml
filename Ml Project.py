# %%
#libraies we need
import numpy as np 
import pandas as pd 
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
#reviewing the data
df=pd.read_csv('train.csv')
l1=df.columns

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('SalePrice',axis=1), df['SalePrice'],                 # Features and target
    test_size=0.2,        # 20% for testing
    random_state=42,      # For reproducibility
)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)
X_train.shape


# %%
#manual classiication of columns based wether the data is numerical,nominal or ordinal data 
numerical=X_train[['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
              'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
              'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
              '3SsnPorch','ScreenPorch','PoolArea','MiscVal']]
nominal=X_train[['Street','Alley','LandContour','LotConfig','Neighborhood','Condition1',
           'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
           'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',
           'GarageType','GarageFinish','MiscFeature','SaleType','SaleCondition']]
ordinal=X_train[['MSSubClass','MSZoning','LotShape','Utilities','LandSlope','OverallQual','OverallCond'
           ,'YearBuilt','YearRemodAdd','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
           'HeatingQC','Electrical','KitchenQual','Functional','FireplaceQu','GarageYrBlt','GarageFinish','GarageQual','GarageCond',
           'PavedDrive','PoolQC','Fence','MoSold','YrSold']]

# %%
#Functions to check for dominant columns and excecive nans in order to remove as they are not necessary for the model
def get_low_information_columns(df, dominance_threshold=0.95):
    """
    Finds columns where one category dominates (occurs in >= dominance_threshold proportion of rows).
    Parameters:
    - df: pandas DataFrame
    - column_list: list of columns to check
    - dominance_threshold: float (0 to 1), threshold for dominant category
    
    Returns:
    - List of column names with dominant single-category values
    """
    low_info_cols = []

    for col in df.columns:
        top_freq = df[col].value_counts(normalize=True, dropna=False).values[0]
        if top_freq >= dominance_threshold:
            low_info_cols.append(col)

    return low_info_cols

def get_columns_with_excessive_nans(df, threshold=0.5):
    """
    Returns columns from column_list where the proportion of NaN values exceeds the given threshold.
    Parameters:
    - df: pandas DataFrame
    - column_list: list of column names to check
    - threshold: float between 0 and 1 (e.g. 0.5 means drop if >50% NaNs)
    Returns:
    - List of column names to consider dropping
    """
    total_rows = len(df)
    drop_candidates = []
    for col in df.columns:
        nan_ratio = df[col].isna().sum() / total_rows
        if nan_ratio > threshold:
            drop_candidates.append(col)
    return drop_candidates


# %%
#selecting columns to drop based on their dominance and nans number

numerical_drop=['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','BedroomAbvGr','GarageCars']

ordinal_drops=get_low_information_columns(ordinal,0.7)
nominal_drops=get_low_information_columns(nominal,0.7)

nan_numerical=get_columns_with_excessive_nans(numerical,0.7)
nan_nominal=get_columns_with_excessive_nans(nominal,0.7)
nan_ordinal=get_columns_with_excessive_nans(ordinal,0.7)

tot_num_drop=list(set(numerical_drop+nan_numerical))
tot_nom_drop=list(set(nominal_drops+nan_nominal))
tot_ord_drop=list(set(ordinal_drops+nan_ordinal))
tot_num_drop
#dropat

numerical_v2=numerical.drop(tot_num_drop,axis=1)
nominal_v2=nominal.drop(tot_nom_drop,axis=1)
ordinal_v2=ordinal.drop(tot_ord_drop,axis=1)
ordinal_v2=ordinal_v2.drop('MoSold',axis=1)
ordinal_v2.shape


# %%
ordinal_drops

# %%
#feature engineering
full_bath=['BsmtFullBath','FullBath']
half_bath=['BsmtHalfBath','HalfBath']
tot_area=['LotFrontage','LotArea','MasVnrArea','TotalBsmtSF','1stFlrSF','2ndFlrSF','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch',
          'PoolArea']
#Adding all th area columns of the house into a single area of the house
numerical_v2['total_area']=numerical_v2[tot_area].sum(axis=1)
numerical_v2.drop(tot_area,axis=1,inplace=True)
#Adding all the full bathrooms and half bathrooms into a single column
numerical_v2['totalbathreoams']=numerical_v2[full_bath].sum(axis=1)+(numerical_v2[half_bath].sum(axis=1)*0.5)
numerical_v2=numerical_v2.drop(full_bath,axis=1)
numerical_v2=numerical_v2.drop(half_bath,axis=1)


significant=['LowQualFinSF','GrLivArea','KitchenAbvGr','TotRmsAbvGrd']#expected signeficant columns from data understanding
numerical_v2.shape

# %%
# the correlation matrix
corr_matrix = numerical_v2.corr(numeric_only=True)
# Convert it to long format for Plotly
corr_long = corr_matrix.reset_index().melt(id_vars='index')
corr_long.columns = ['Feature1', 'Feature2', 'Correlation']
# Create Plotly heatmap
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
fig.show()

# %%
numerical_v2=numerical_v2.drop('MiscVal',axis=1)
numerical_v2=numerical_v2.drop('GrLivArea',axis=1)
numerical_v2

# %%
#nominal encoding 
#nominal encoding 
encoded_nominal=pd.DataFrame()
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# Fit and transform only the specified columns
encoded_data = ohe.fit_transform(nominal_v2)

# Get the new column names for encoded features
encoded_cols = ohe.get_feature_names_out(nominal_v2.columns)

# Create a DataFrame with the encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=nominal_v2.index)

# Drop original columns and concatenate the encoded ones

encoded_nominal = pd.concat([encoded_nominal, encoded_df], axis=1)
encoded_nominal.shape


# %%
#ordinal encoding
#small function to show all the unique values of each column (numerical are sorted) in order to make the order manually for the encoder 
'''for col in ordinal_v2.columns:
    uniques = sorted(ordinal_v2[col].dropna().unique())

    cleaned_uniques = []
    for val in uniques:
        if isinstance(val, (np.integer, int)):
            cleaned_uniques.append(int(val))
        elif isinstance(val, (np.floating, float)):
            # Convert float to int if it's a whole number like 1900.0
            cleaned_uniques.append(int(val) if val.is_integer() else float(val))
        else:
            cleaned_uniques.append(val)

    print(f"{col}: {cleaned_uniques}")
'''


orders_list=[[20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190],
             ['Reg','IR1', 'IR2', 'IR3'],
             [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
             [1, 2, 3, 4, 5, 6, 7, 8, 9],
             [1872, 1875, 1880, 1882, 1885, 1890, 1892, 1893, 1898, 1900, 1904, 1905, 1906, 1908,
              1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917,1918, 1919, 1920, 1921, 1922, 1923, 
              1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1934, 1935, 1936, 1937, 1938,
              1939, 1940,1941, 1942, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 
              1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964,1965, 1966, 1967, 1968, 
              1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982,
              1983, 1984, 1985,1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 
              1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
             [1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 
              1964, 1965, 1966, 1967, 1968, 1969, 1970,1971, 1972, 1973, 1974, 1975, 1976, 1977, 
              1978, 1979, 1980, 1981, 1982, 1983,1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 
              1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005,
              2006, 2007, 2008, 2009, 2010],
             ['Ex', 'Gd', 'TA', 'Fa'],
             ['Ex', 'Gd', 'TA', 'Fa'],
             [ 'Gd','Av', 'Mn', 'No'],
             ['GLQ','ALQ', 'BLQ', 'Rec', 'LwQ',  'Unf'],
             ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
             ['Ex', 'Gd', 'TA', 'Fa'],
             ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
             [1900, 1906, 1908, 1910, 1914, 1915, 1916, 1918, 1920, 1921, 1922, 1923, 1924, 1925,
              1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934, 1935, 1936, 1937, 1938, 1939,
              1940, 1941, 1942, 1945, 1946, 1947, 1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955,
              1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969,
              1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983,
              1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
              1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010],
              ['Fin', 'RFn', 'Unf'],
               [2006, 2007, 2008, 2009, 2010]]

ordinal_cols=ordinal_v2.columns
encoder = OrdinalEncoder(categories=orders_list,handle_unknown='use_encoded_value', unknown_value=-1)
ordinal_v2[ordinal_cols] = encoder.fit_transform(ordinal_v2[ordinal_cols])
ordinal_encoded=ordinal_v2
ordinal_encoded.shape



# %%
#final data frame (nesceds scaling and ready for ML)
numerical_v2
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

numerical_v3=scaler.fit_transform(numerical_v2)
numerical_v4=pd.DataFrame(numerical_v3,columns=numerical_v2.columns)
final_train=pd.concat([numerical_v4,encoded_nominal,ordinal_encoded],axis=1)
final_train.shape



# %%
#encapsulate in a pipeline in the future 

df=pd.read_csv('test.csv')
l1=df.columns
df['1stFlrSF']


# %%
#manual classiication of columns based wether the data is numerical,nominal or ordinal data 
numerical_test=X_test[['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF',
              'LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
              'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch',
              '3SsnPorch','ScreenPorch','PoolArea','MiscVal']]
nominal_test=X_test[['Street','Alley','LandContour','LotConfig','Neighborhood','Condition1',
           'Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
           'Exterior1st','Exterior2nd','MasVnrType','Foundation','Heating','CentralAir',
           'GarageType','GarageFinish','MiscFeature','SaleType','SaleCondition']]
ordinal_test=X_test[['MSSubClass','MSZoning','LotShape','Utilities','LandSlope','OverallQual','OverallCond'
           ,'YearBuilt','YearRemodAdd','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
           'HeatingQC','Electrical','KitchenQual','Functional','FireplaceQu','GarageYrBlt','GarageFinish','GarageQual','GarageCond',
           'PavedDrive','PoolQC','Fence','MoSold','YrSold']]

'''ordinal_v2.shape


ordinal_test=ordinal_test.drop(tot_ord_drop,axis=1)

ordinal_test.drop('MoSold',axis=1,inplace=True)
ordinal_v2.shape
ordinal_test.shape
numerical_test=numerical_test.drop(tot_num_drop,axis=1)
numerical_test.shape
numerical_v2
numerical_test.head()'''

numerical_test=numerical_test.drop(tot_num_drop,axis=1)
nominal_test=nominal_test.drop(tot_nom_drop,axis=1)
ordinal_test=ordinal_test.drop(tot_ord_drop,axis=1)
ordinal_test=ordinal_test.drop('MoSold',axis=1)

#feature engineering

#Adding all th area columns of the house into a single area of the house
numerical_test['total_area']=numerical_test[tot_area].sum(axis=1)
numerical_test.drop(tot_area,axis=1,inplace=True)
#Adding all the full bathrooms and half bathrooms into a single column
numerical_test['totalbathreoams']=numerical_test[full_bath].sum(axis=1)+(numerical_test[half_bath].sum(axis=1)*0.5)
numerical_test=numerical_test.drop(full_bath,axis=1)
numerical_test=numerical_test.drop(half_bath,axis=1)
numerical_test=numerical_test.drop('MiscVal',axis=1)
numerical_test=numerical_test.drop('GrLivArea',axis=1)

#nominal encoding 
encoded_nominal_test=pd.DataFrame()
 

# Fit and transform only the specified columns
encoded_data = ohe.transform(nominal_test)

# Get the new column names for encoded features
encoded_cols = ohe.get_feature_names_out(nominal_test.columns)

# Create a DataFrame with the encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=encoded_cols, index=nominal_test.index)

# Drop original columns and concatenate the encoded ones

encoded_nominal_test = pd.concat([encoded_nominal_test, encoded_df], axis=1)


ordinal_cols=ordinal_test.columns
ordinal_test[ordinal_cols] = encoder.transform(ordinal_test[ordinal_cols])
ordinal_encoded_test=ordinal_test
ordinal_encoded_test


ordinal_encoded_test.shape


numerical_test
from sklearn.preprocessing import StandardScaler

numerical_test_v3=scaler.fit_transform(numerical_test)
numerical_test_v4=pd.DataFrame(numerical_test_v3,columns=numerical_test.columns)
final_test=pd.concat([numerical_test_v4,encoded_nominal_test,ordinal_encoded_test],axis=1)
y_train.shape



X_train=final_train
X_test=final_test



# %%
#First try using sbase reggresors with average majority voting ensemble
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score

# Define base regressors
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
svr = SVR()

# Train them
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
svr.fit(X_train, y_train)

# Predict
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
svr_pred = svr.predict(X_test)

# Simple average
avg_pred = (lr_pred + rf_pred + svr_pred) / 3

# Evaluate
mse = mean_squared_error(y_test, avg_pred)
r2 = r2_score(y_test, avg_pred)

print(f"Simple Averaging MSE: {mse:.2f}")
print(f"Simple Averaging R² Score: {r2:.4f}")

# %%
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=y_test, y=avg_pred,
                         mode='markers',
                         name='Predicted vs Actual',
                         marker=dict(color='blue', opacity=0.6)))

# Line of perfect prediction
fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                         y=[y_test.min(), y_test.max()],
                         mode='lines',
                         name='Perfect Prediction',
                         line=dict(color='red', dash='dash')))

fig.update_layout(title="Actual vs Predicted (Averaged Ensemble)",
                  xaxis_title="Actual",
                  yaxis_title="Predicted",
                  template="plotly_white")
fig.show()

# %%
residuals = y_test - avg_pred

fig = go.Figure()

fig.add_trace(go.Scatter(x=avg_pred, y=residuals,
                         mode='markers',
                         name='Residuals',
                         marker=dict(color='orange', opacity=0.6)))

fig.add_hline(y=0, line_dash="dash", line_color="red")

fig.update_layout(title="Residual Plot",
                  xaxis_title="Predicted",
                  yaxis_title="Residuals",
                  template="plotly_white")
fig.show()

# %%
models = ['Linear Regression', 'Random Forest', 'SVR', 'Ensemble Average']
mses = [
    mean_squared_error(y_test, lr_pred),
    mean_squared_error(y_test, rf_pred),
    mean_squared_error(y_test, svr_pred),
    mean_squared_error(y_test, avg_pred)
]

fig = go.Figure([go.Bar(x=models, y=mses, marker_color=['skyblue', 'lightgreen', 'salmon', 'purple'])])

fig.update_layout(title="Model MSE Comparison",
                  xaxis_title="Model",
                  yaxis_title="MSE",
                  template="plotly_white")
fig.show()

# %%
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor,early_stopping, log_evaluation

# Initialize models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(random_state=42, verbosity=0)
lgb = LGBMRegressor(random_state=42)

# Fit models
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lgb.fit(X_train, y_train)

# Predict on test set
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)
lgb_pred = lgb.predict(X_test)

# Average predictions
avg_pred = (lr_pred + rf_pred + xgb_pred + lgb_pred) / 4

# Evaluation
mse = mean_squared_error(y_test, avg_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, avg_pred)

print(f"Simple Averaging MSE: {mse:.2f}")
print(f"Simple Averaging RMSE: {rmse:.2f}")
print(f"Simple Averaging R² Score: {r2:.4f}")


# %%
models = ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM', 'Ensemble Avg']
mses = [
    mean_squared_error(y_test, lr_pred),
    mean_squared_error(y_test, rf_pred),
    mean_squared_error(y_test, xgb_pred),
    mean_squared_error(y_test, lgb_pred),
    mean_squared_error(y_test, avg_pred)
]

fig = go.Figure([go.Bar(x=models, y=mses, marker_color=['skyblue', 'lightgreen', 'goldenrod', 'darkgreen', 'purple'])])

fig.update_layout(title="Model MSE Comparison",
                  xaxis_title="Model",
                  yaxis_title="MSE",
                  template="plotly_white")
fig.show()

# %%
xgb_importance = xgb.feature_importances_
features = final_train.columns

fig = go.Figure([go.Bar(x=features, y=xgb_importance, marker_color='dodgerblue')])
fig.update_layout(title="XGBoost Feature Importance",
                  xaxis_title="Feature",
                  yaxis_title="Importance",
                  template="plotly_white")
fig.show()

# %%
lgb_importance = lgb.feature_importances_

fig = go.Figure([go.Bar(x=features, y=lgb_importance, marker_color='forestgreen')])
fig.update_layout(title="LightGBM Feature Importance",
                  xaxis_title="Feature",
                  yaxis_title="Importance",
                  template="plotly_white")
fig.show()

# %%
# Example: you already have these predictions from your trained models on X_test
# lr_pred = lr.predict(X_test_encoded)
# rf_pred = rf.predict(X_test_encoded)
# xgb_pred = xgb.predict(X_test_encoded)
# lgb_pred = lgb.predict(X_test_encoded)

# For demonstration, I will assume lr_pred, rf_pred, xgb_pred, lgb_pred are numpy arrays
# y_test is the true target values
from scipy.optimize import minimize

def objective(weights):
    # Weighted sum of predictions
    weighted_pred = (
        weights[0] * lr_pred +
        weights[1] * rf_pred +
        weights[2] * xgb_pred +
        weights[3] * lgb_pred
    )
    # We want to maximize R2, so minimize negative R2
    return -r2_score(y_test, weighted_pred)

# Constraint: weights must sum to 1
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# Bounds: each weight between 0 and 1
bounds = [(0,1)] * 4

# Initial guess: equal weights
init_guess = [0.25, 0.25, 0.25, 0.25]

result = minimize(objective, init_guess, bounds=bounds, constraints=constraints)

best_weights = result.x
best_r2 = -result.fun

print(f"Optimized weights: {best_weights}")
print(f"Optimized R² score: {best_r2:.4f}")

# Calculate final weighted prediction with optimized weights
final_pred = (
    best_weights[0] * lr_pred +
    best_weights[1] * rf_pred +
    best_weights[2] * xgb_pred +
    best_weights[3] * lgb_pred
)

# Optionally calculate MSE and RMSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, final_pred)
rmse = np.sqrt(mse)

print(f"Weighted Averaging MSE: {mse:.2f}")
print(f"Weighted Averaging RMSE: {rmse:.2f}")

# %%
model_names = ['Linear Regression', 'Random Forest', 'XGBoost', 'LightGBM']

fig = go.Figure([go.Bar(x=model_names, y=best_weights,
                        marker_color=['skyblue', 'lightgreen', 'goldenrod', 'forestgreen'])])

fig.update_layout(title="Optimized Model Weights (Weighted Ensemble)",
                  xaxis_title="Model",
                  yaxis_title="Weight",
                  template="plotly_white")
fig.show()

# %%
fig = px.histogram(residuals, nbins=30, marginal='box',
                   title="Residual Distribution (Weighted Ensemble)",
                   labels={'value': 'Prediction Error'},
                   template="plotly_white")

fig.update_layout(xaxis_title="Residual", yaxis_title="Frequency")
fig.show()

# %%
# Assuming avg_pred (simple average) and final_pred (weighted) exist
from sklearn.metrics import r2_score, mean_squared_error

compare_df = {
    'Model': ['Simple Avg', 'Weighted Avg'],
    'MSE': [mean_squared_error(y_test, avg_pred), mean_squared_error(y_test, final_pred)],
    'R2': [r2_score(y_test, avg_pred), r2_score(y_test, final_pred)]
}

import pandas as pd
df_compare = pd.DataFrame(compare_df)

fig = go.Figure()
fig.add_trace(go.Bar(x=df_compare['Model'], y=df_compare['MSE'], name='MSE', marker_color='lightsalmon'))
fig.add_trace(go.Bar(x=df_compare['Model'], y=df_compare['R2'], name='R² Score', marker_color='lightblue', yaxis='y2'))

fig.update_layout(
    title="Simple vs Weighted Ensemble Performance",
    xaxis_title="Model",
    yaxis=dict(title='MSE', side='left'),
    yaxis2=dict(title='R² Score', overlaying='y', side='right'),
    barmode='group',
    template='plotly_white'
)
fig.show()

# %%
xgb = XGBRegressor(random_state=42, verbosity=0, eval_metric='rmse')

xgb.fit(
    final_train, y_train,
    eval_set=[(final_train, y_train), (final_test, y_test)],
    verbose=False
)
from lightgbm import early_stopping, log_evaluation

lgb = LGBMRegressor(random_state=42)

lgb.fit(
    final_train, y_train,
    eval_set=[(final_train, y_train), (final_test, y_test)],
    eval_metric='rmse',
    callbacks=[log_evaluation(period=0)]  # disables logging
)

# %%
xgb_eval = xgb.evals_result()
lgb_eval = lgb.evals_result_

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(y=xgb_eval['validation_0']['rmse'], name='XGB Train RMSE'))
fig.add_trace(go.Scatter(y=xgb_eval['validation_1']['rmse'], name='XGB Test RMSE'))

fig.update_layout(title="XGBoost Learning Curve (RMSE)",
                  xaxis_title="Boosting Round",
                  yaxis_title="RMSE",
                  template="plotly_white")
fig.show()

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(y=lgb_eval['training']['rmse'], name='LGBM Train RMSE'))
fig.add_trace(go.Scatter(y=lgb_eval['valid_1']['rmse'], name='LGBM Test RMSE'))

fig.update_layout(title="LightGBM Learning Curve (RMSE)",
                  xaxis_title="Boosting Round",
                  yaxis_title="RMSE",
                  template="plotly_white")
fig.show()

# %%
print(lgb_eval['training'].keys())  # Confirm available metrics under 'training'

# %%
#bagging regressors
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Initialize base model (Decision Tree)
base_model = DecisionTreeRegressor(random_state=42)

# Initialize Bagging Regressor with base model
bagging = BaggingRegressor(estimator=base_model,
                           n_estimators=50,
                           random_state=42)

# Fit on training data
bagging.fit(final_train, y_train)

# Predict on test set
y_pred = bagging.predict(final_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Bagging Regressor MSE: {mse:.2f}")
print(f"Bagging Regressor RMSE: {rmse:.2f}")
print(f"Bagging Regressor R² Score: {r2:.4f}")


# %%
#voting reggresor
from sklearn.ensemble import VotingRegressor
# Initialize individual models
lr = LinearRegression()
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)
lgb = LGBMRegressor(random_state=42)

# Initialize Voting Regressor
voting_reg = VotingRegressor(estimators=[
    ('lr', lr),
    ('rf', rf),
    ('xgb', xgb),
    ('lgb', lgb)
])

# Fit Voting Regressor on training data
voting_reg.fit(final_train, y_train)

# Predict on test data
y_pred = voting_reg.predict(final_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Voting Regressor MSE: {mse:.2f}")
print(f"Voting Regressor RMSE: {rmse:.2f}")
print(f"Voting Regressor R² Score: {r2:.4f}")

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(y=y_test, mode='lines+markers', name='True Values'))
fig.add_trace(go.Scatter(y=voting_reg.predict(final_test), mode='lines+markers', name='Voting Regressor'))

fig.update_layout(title="True vs Voting Regressor Predictions",
                  xaxis_title="Test Sample Index",
                  yaxis_title="Target Value",
                  template="plotly_white")
fig.show()

# %%
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
# Base models
estimators = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('xgb', XGBRegressor(random_state=42)),
    ('lgb', LGBMRegressor(random_state=42))
]

# Meta-model: Ridge Regression (simple, effective)
meta_regressor = Ridge(alpha=1.0)

# Initialize Stacking with cv=10
stacking_reg = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_regressor,
    cv=10,
    n_jobs=-1
)

# Train
stacking_reg.fit(final_train, y_train)

# Predict
y_pred = stacking_reg.predict(final_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Stacking Regressor with cv=10 MSE: {mse:.2f}")
print(f"Stacking Regressor with cv=10 RMSE: {rmse:.2f}")
print(f"Stacking Regressor with cv=10 R² Score: {r2:.4f}")

# %%
residuals = y_test - y_pred

fig = go.Figure()
fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals'))

fig.update_layout(title="Residuals Distribution of Stacking Regressor",
                  xaxis_title="Residual",
                  yaxis_title="Frequency",
                  template="plotly_white")
fig.show()

# %%
feature_names = [f"Feature {i}" for i in range(final_train.shape[1])]

# Initialize base models with the same random states as in stacking
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(random_state=42, verbosity=0)
lgb = LGBMRegressor(random_state=42)

# Fit each model
rf.fit(final_train, y_train)
xgb.fit(final_train, y_train)
lgb.fit(final_train, y_train)

# Extract feature importances
rf_importances = rf.feature_importances_
xgb_importances = xgb.feature_importances_
lgb_importances = lgb.feature_importances_

# %%
fig = go.Figure()

fig.add_trace(go.Bar(
    x=feature_names,
    y=rf_importances,
    name="Random Forest"
))

fig.add_trace(go.Bar(
    x=feature_names,
    y=xgb_importances,
    name="XGBoost"
))

fig.add_trace(go.Bar(
    x=feature_names,
    y=lgb_importances,
    name="LightGBM"
))

fig.update_layout(
    title="Feature Importances from Base Models",
    xaxis_title="Features",
    yaxis_title="Importance",
    barmode='group',
    template="plotly_white",
    xaxis_tickangle=-45,
    height=500
)

fig.show()

# %%
# Step 1: Split your existing training data into base training and holdout (for blending)
X_base_train, X_holdout, y_base_train, y_holdout = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Step 2: Initialize base models
rf = RandomForestRegressor(random_state=42)
xgb = XGBRegressor(random_state=42)
lgb = LGBMRegressor(random_state=42)

# Step 3: Train base models on the base training set only
rf.fit(X_base_train, y_base_train)
xgb.fit(X_base_train, y_base_train)
lgb.fit(X_base_train, y_base_train)

# Step 4: Generate predictions of base models on the holdout set (meta-model training data)
rf_holdout_pred = rf.predict(X_holdout)
xgb_holdout_pred = xgb.predict(X_holdout)
lgb_holdout_pred = lgb.predict(X_holdout)

# Step 5: Stack these predictions as features to train the meta-model (blender)
X_blend_train = np.column_stack((rf_holdout_pred, xgb_holdout_pred, lgb_holdout_pred))

# Step 6: Initialize and train meta-model (e.g., Ridge)
meta_model = Ridge(alpha=1.0)
meta_model.fit(X_blend_train, y_holdout)

# Step 7: Predict base models on the original test set
rf_test_pred = rf.predict(X_test)
xgb_test_pred = xgb.predict(X_test)
lgb_test_pred = lgb.predict(X_test)

# Step 8: Stack test predictions as input for meta-model
X_blend_test = np.column_stack((rf_test_pred, xgb_test_pred, lgb_test_pred))

# Step 9: Make final blended predictions
y_pred = meta_model.predict(X_blend_test)

# Step 10: Evaluate the blended predictions
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Blending MSE: {mse:.2f}")
print(f"Blending RMSE: {rmse:.2f}")
print(f"Blending R² Score: {r2:.4f}")

# %%
preds_df = pd.DataFrame({
    'RandomForest': rf_test_pred,
    'XGBoost': xgb_test_pred,
    'LightGBM': lgb_test_pred
})

fig = px.imshow(preds_df.corr(), 
                text_auto=True, 
                color_continuous_scale='Viridis', 
                title='Correlation Matrix of Base Model Predictions')
fig.show()

# %%
coef_df = pd.DataFrame({
    'Model': ['RandomForest', 'XGBoost', 'LightGBM'],
    'Coefficient': meta_model.coef_
})

fig = px.bar(coef_df, x='Model', y='Coefficient',
             title='Meta-Model Coefficients (Blender Weights)',
             text='Coefficient')
fig.show()


# %%
errors_df = pd.DataFrame({
    'RandomForest Error': np.abs(y_test - rf_test_pred),
    'XGBoost Error': np.abs(y_test - xgb_test_pred),
    'LightGBM Error': np.abs(y_test - lgb_test_pred),
    'Blender Error': np.abs(y_test - y_pred)
})

fig = go.Figure()
for col in errors_df.columns:
    fig.add_trace(go.Box(y=errors_df[col], name=col))

fig.update_layout(title='Prediction Errors Distribution by Model',
                  yaxis_title='Absolute Error',
                  template='plotly_white')
fig.show()

# %%
best_base_error = np.minimum.reduce([
    np.abs(y_test - rf_test_pred),
    np.abs(y_test - xgb_test_pred),
    np.abs(y_test - lgb_test_pred)
])

improvement = best_base_error - np.abs(y_test - y_pred)  # Positive means blender better

fig = px.histogram(improvement, nbins=30, 
                   title='Error Improvement by Blending over Best Base Model',
                   labels={'value':'Improvement in Absolute Error'})
fig.show()

# %%
fig = px.scatter(x=rf_test_pred, y=xgb_test_pred,
                 labels={'x':'RandomForest Predictions', 'y':'XGBoost Predictions'},
                 title='Scatter of RF vs XGB Predictions')
fig.show()

# %%
import optuna
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Re-split to ensure clean training/validation split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 2),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 2),
        "random_state": 42,
    }

    model = XGBRegressor(**params)
    model.fit(X_tr, y_tr)  # removed early stopping
    preds = model.predict(X_val)
    return r2_score(y_val, preds)

# Optimize
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best hyperparameters:", study.best_trial.params)

# %%
import optuna.visualization as vis

# 1. Optimization history (objective value per trial)
fig1 = vis.plot_optimization_history(study)
fig1.show()

# 2. Hyperparameter importance
fig2 = vis.plot_param_importances(study)
fig2.show()

# 3. Parallel coordinate plot to see hyperparameter effects together
fig3 = vis.plot_parallel_coordinate(study)
fig3.show()

# 4. Contour plot for interactions between hyperparameters
fig4 = vis.plot_contour(study)
fig4.show()


# %%
# Use best parameters from Optuna study
best_params = study.best_trial.params

# You may want to re-specify fixed params like random_state
best_params["random_state"] = 42

# Train final model on full training set
final_xgb_model = XGBRegressor(**best_params)
final_xgb_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = final_xgb_model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error
print("Optimized R² Score:", r2_score(y_test, y_pred))
print("Optimized MSE:", mean_squared_error(y_test, y_pred))


# %%
#catboost
from catboost import CatBoostRegressor
# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Optuna objective
def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 100, 1000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10),
        "loss_function": "RMSE",
        "verbose": 0,
        "random_state": 42
    }

    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=0)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30, timeout=600)

# Train final model
best_params = study.best_params
final_model = CatBoostRegressor(**best_params, loss_function="RMSE", verbose=0, random_state=42)
final_model.fit(X_train_val, y_train_val)

# Evaluate
y_pred = final_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Parameters:", best_params)
print("MSE:", mse)
print("R² Score:", r2)


# %%
# Extract trial numbers and their corresponding values (validation MSE)
trial_nums = [trial.number for trial in study.trials]
mse_values = [trial.value for trial in study.trials]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=trial_nums,
    y=mse_values,
    mode='lines+markers',
    name='Validation MSE',
    line=dict(color='blue'),
    marker=dict(size=8)
))

fig.update_layout(
    title='Optuna Optimization Progress',
    xaxis_title='Trial Number',
    yaxis_title='Validation MSE',
    template='plotly_white'
)

fig.show()

# %%
learning_rates = [trial.params.get("learning_rate", None) for trial in study.trials]

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=trial_nums,
    y=learning_rates,
    mode='markers',
    name='Learning Rate',
    marker=dict(color='green', size=8)
))

fig2.update_layout(
    title='Learning Rate Across Trials',
    xaxis_title='Trial Number',
    yaxis_title='Learning Rate',
    template='plotly_white'
)

fig2.show()


