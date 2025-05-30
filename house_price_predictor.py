import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, VotingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna

class HousePricePredictor:
    def __init__(self, train_path='train.csv', test_path='test.csv'):
        self.train_path = train_path
        self.test_path = test_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.final_train = None
        self.final_test = None
        self.numerical = None
        self.nominal = None
        self.ordinal = None
        self.encoders = {}
        self.scalers = {}
        self.optimization_history = None
        
    def load_data(self):
        """Load and split the data"""
        self.df = pd.read_csv(self.train_path)
        X_train, X_test, y_train, y_test = train_test_split(
            self.df.drop('SalePrice', axis=1),
            self.df['SalePrice'],
            test_size=0.2,
            random_state=42
        )
        self.X_train = X_train.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        
    def classify_columns(self):
        """Classify columns into numerical, nominal, and ordinal"""
        self.numerical = self.X_train[['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                                     'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                                     'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
                                     'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                                     'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                                     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']]
        
        self.nominal = self.X_train[['Street', 'Alley', 'LandContour', 'LotConfig', 'Neighborhood',
                                   'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                                   'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
                                   'Heating', 'CentralAir', 'GarageType', 'GarageFinish', 'MiscFeature',
                                   'SaleType', 'SaleCondition']]
        
        self.ordinal = self.X_train[['MSSubClass', 'MSZoning', 'LotShape', 'Utilities', 'LandSlope',
                                   'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                                   'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                                   'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical',
                                   'KitchenQual', 'Functional', 'FireplaceQu', 'GarageYrBlt',
                                   'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                                   'PoolQC', 'Fence', 'MoSold', 'YrSold']]

    def get_low_information_columns(self, df, dominance_threshold=0.95):
        """Find columns with dominant single-category values"""
        low_info_cols = []
        for col in df.columns:
            top_freq = df[col].value_counts(normalize=True, dropna=False).values[0]
            if top_freq >= dominance_threshold:
                low_info_cols.append(col)
        return low_info_cols

    def get_columns_with_excessive_nans(self, df, threshold=0.5):
        """Find columns with excessive NaN values"""
        total_rows = len(df)
        drop_candidates = []
        for col in df.columns:
            nan_ratio = df[col].isna().sum() / total_rows
            if nan_ratio > threshold:
                drop_candidates.append(col)
        return drop_candidates

    def preprocess_data(self):
        """Preprocess the data including feature engineering and encoding"""
        # Drop columns based on dominance and NaN values
        numerical_drop = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BedroomAbvGr', 'GarageCars']
        ordinal_drops = self.get_low_information_columns(self.ordinal, 0.7)
        nominal_drops = self.get_low_information_columns(self.nominal, 0.7)
        
        nan_numerical = self.get_columns_with_excessive_nans(self.numerical, 0.7)
        nan_nominal = self.get_columns_with_excessive_nans(self.nominal, 0.7)
        nan_ordinal = self.get_columns_with_excessive_nans(self.ordinal, 0.7)
        
        tot_num_drop = list(set(numerical_drop + nan_numerical))
        tot_nom_drop = list(set(nominal_drops + nan_nominal))
        tot_ord_drop = list(set(ordinal_drops + nan_ordinal))
        
        # Drop columns
        numerical_v2 = self.numerical.drop(tot_num_drop, axis=1)
        nominal_v2 = self.nominal.drop(tot_nom_drop, axis=1)
        ordinal_v2 = self.ordinal.drop(tot_ord_drop, axis=1)
        ordinal_v2 = ordinal_v2.drop('MoSold', axis=1)
        
        # Feature engineering
        full_bath = ['BsmtFullBath', 'FullBath']
        half_bath = ['BsmtHalfBath', 'HalfBath']
        tot_area = ['LotFrontage', 'LotArea', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                   'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                   'ScreenPorch', 'PoolArea']
        
        numerical_v2['total_area'] = numerical_v2[tot_area].sum(axis=1)
        numerical_v2.drop(tot_area, axis=1, inplace=True)
        numerical_v2['totalbathreoams'] = (numerical_v2[full_bath].sum(axis=1) + 
                                         (numerical_v2[half_bath].sum(axis=1) * 0.5))
        numerical_v2 = numerical_v2.drop(full_bath + half_bath, axis=1)
        numerical_v2 = numerical_v2.drop(['MiscVal', 'GrLivArea'], axis=1)
        
        # Encoding
        # Nominal encoding
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_data = ohe.fit_transform(nominal_v2)
        encoded_cols = ohe.get_feature_names_out(nominal_v2.columns)
        encoded_nominal = pd.DataFrame(encoded_data, columns=encoded_cols, index=nominal_v2.index)
        self.encoders['ohe'] = ohe
        
        # Ordinal encoding
        orders_list = self._get_ordinal_orders()
        encoder = OrdinalEncoder(categories=orders_list, handle_unknown='use_encoded_value', unknown_value=-1)
        ordinal_v2[ordinal_v2.columns] = encoder.fit_transform(ordinal_v2[ordinal_v2.columns])
        self.encoders['ordinal'] = encoder
        
        # Scaling
        scaler = StandardScaler()
        numerical_v3 = scaler.fit_transform(numerical_v2)
        numerical_v4 = pd.DataFrame(numerical_v3, columns=numerical_v2.columns)
        self.scalers['numerical'] = scaler
        
        # Combine all features
        self.final_train = pd.concat([numerical_v4, encoded_nominal, ordinal_v2], axis=1)
        
        # Process test data similarly
        self._process_test_data()

    def _get_ordinal_orders(self):
        """Return the predefined order for ordinal encoding"""
        return [
            [20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190],
            ['Reg', 'IR1', 'IR2', 'IR3'],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            list(range(1872, 2011)),
            list(range(1950, 2011)),
            ['Ex', 'Gd', 'TA', 'Fa'],
            ['Ex', 'Gd', 'TA', 'Fa'],
            ['Gd', 'Av', 'Mn', 'No'],
            ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf'],
            ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
            ['Ex', 'Gd', 'TA', 'Fa'],
            ['Ex', 'Gd', 'TA', 'Fa', 'Po'],
            list(range(1900, 2011)),
            ['Fin', 'RFn', 'Unf'],
            list(range(2006, 2011))
        ]

    def _process_test_data(self):
        """Process test data using the same transformations as training data"""
        # Get the same columns as training data
        numerical_test = self.X_test[self.numerical.columns]
        nominal_test = self.X_test[self.nominal.columns]
        ordinal_test = self.X_test[self.ordinal.columns]
        
        # Drop the same columns as in training
        numerical_drop = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BedroomAbvGr', 'GarageCars']
        ordinal_drops = self.get_low_information_columns(self.ordinal, 0.7)
        nominal_drops = self.get_low_information_columns(self.nominal, 0.7)
        
        nan_numerical = self.get_columns_with_excessive_nans(self.numerical, 0.7)
        nan_nominal = self.get_columns_with_excessive_nans(self.nominal, 0.7)
        nan_ordinal = self.get_columns_with_excessive_nans(self.ordinal, 0.7)
        
        tot_num_drop = list(set(numerical_drop + nan_numerical))
        tot_nom_drop = list(set(nominal_drops + nan_nominal))
        tot_ord_drop = list(set(ordinal_drops + nan_ordinal))
        
        # Drop columns
        numerical_v2 = numerical_test.drop(tot_num_drop, axis=1)
        nominal_v2 = nominal_test.drop(tot_nom_drop, axis=1)
        ordinal_v2 = ordinal_test.drop(tot_ord_drop, axis=1)
        ordinal_v2 = ordinal_v2.drop('MoSold', axis=1)
        
        # Feature engineering
        full_bath = ['BsmtFullBath', 'FullBath']
        half_bath = ['BsmtHalfBath', 'HalfBath']
        tot_area = ['LotFrontage', 'LotArea', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                   'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
                   'ScreenPorch', 'PoolArea']
        
        numerical_v2['total_area'] = numerical_v2[tot_area].sum(axis=1)
        numerical_v2.drop(tot_area, axis=1, inplace=True)
        numerical_v2['totalbathreoams'] = (numerical_v2[full_bath].sum(axis=1) + 
                                         (numerical_v2[half_bath].sum(axis=1) * 0.5))
        numerical_v2 = numerical_v2.drop(full_bath + half_bath, axis=1)
        numerical_v2 = numerical_v2.drop(['MiscVal', 'GrLivArea'], axis=1)
        
        # Encoding
        # Nominal encoding
        encoded_data = self.encoders['ohe'].transform(nominal_v2)
        encoded_cols = self.encoders['ohe'].get_feature_names_out(nominal_v2.columns)
        encoded_nominal = pd.DataFrame(encoded_data, columns=encoded_cols, index=nominal_v2.index)
        
        # Ordinal encoding
        ordinal_v2[ordinal_v2.columns] = self.encoders['ordinal'].transform(ordinal_v2[ordinal_v2.columns])
        
        # Scaling
        numerical_v3 = self.scalers['numerical'].transform(numerical_v2)
        numerical_v4 = pd.DataFrame(numerical_v3, columns=numerical_v2.columns)
        
        # Combine all features
        self.final_test = pd.concat([numerical_v4, encoded_nominal, ordinal_v2], axis=1)

    def train_models(self):
        """Train various models and create ensembles"""
        # Initialize base models
        self.models = {
            'lr': LinearRegression(),
            'rf': RandomForestRegressor(random_state=42),
            'xgb': XGBRegressor(random_state=42),
            'lgb': LGBMRegressor(random_state=42)
        }
        
        # Train base models
        for name, model in self.models.items():
            model.fit(self.final_train, self.y_train)
        
        # Create and train ensembles
        self._create_ensembles()
        
    def _create_ensembles(self):
        """Create and train various ensemble methods"""
        # Voting Regressor
        self.voting_reg = VotingRegressor(estimators=[
            ('lr', self.models['lr']),
            ('rf', self.models['rf']),
            ('xgb', self.models['xgb']),
            ('lgb', self.models['lgb'])
        ])
        self.voting_reg.fit(self.final_train, self.y_train)
        
        # Stacking Regressor
        self.stacking_reg = StackingRegressor(
            estimators=[
                ('rf', self.models['rf']),
                ('xgb', self.models['xgb']),
                ('lgb', self.models['lgb'])
            ],
            final_estimator=Ridge(alpha=1.0),
            cv=10
        )
        self.stacking_reg.fit(self.final_train, self.y_train)

    def evaluate_models(self):
        """Evaluate all models and create visualizations"""
        results = {}
        
        # Evaluate base models
        for name, model in self.models.items():
            y_pred = model.predict(self.final_test)
            results[name] = {
                'mse': mean_squared_error(self.y_test, y_pred),
                'r2': r2_score(self.y_test, y_pred)
            }
        
        # Evaluate ensembles
        voting_pred = self.voting_reg.predict(self.final_test)
        stacking_pred = self.stacking_reg.predict(self.final_test)
        
        results['voting'] = {
            'mse': mean_squared_error(self.y_test, voting_pred),
            'r2': r2_score(self.y_test, voting_pred)
        }
        
        results['stacking'] = {
            'mse': mean_squared_error(self.y_test, stacking_pred),
            'r2': r2_score(self.y_test, stacking_pred)
        }
        
        return results

    def optimize_hyperparameters(self, model_name='xgb', n_trials=50):
        """Optimize hyperparameters using Optuna"""
        if model_name == 'xgb':
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
                model.fit(self.final_train, self.y_train)
                preds = model.predict(self.final_test)
                return mean_squared_error(self.y_test, preds)
        
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        
        # Store optimization history
        self.optimization_history = {
            'trial_numbers': [trial.number for trial in study.trials],
            'values': [trial.value for trial in study.trials],
            'params': [trial.params for trial in study.trials]
        }
        
        return study.best_params

    def get_optimization_history(self):
        """Return the optimization history for plotting"""
        if self.optimization_history is None:
            return None
        return pd.DataFrame({
            'Trial': self.optimization_history['trial_numbers'],
            'MSE': self.optimization_history['values'],
            **{k: [p.get(k) for p in self.optimization_history['params']] 
               for k in self.optimization_history['params'][0].keys()}
        })

    def run_pipeline(self):
        """Run the complete pipeline"""
        self.load_data()
        self.classify_columns()
        self.preprocess_data()
        self.train_models()
        results = self.evaluate_models()
        best_params = self.optimize_hyperparameters()
        
        return results, best_params 