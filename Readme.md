# ML Workflow App

A modern, professional Streamlit app for end-to-end regression ML workflows, focused on ensemble modeling and visualizations.

## Features
- **Automated Data Loading & Preprocessing**: Loads `train.csv` on startup, applies preprocessing and feature engineering using `preprocessing.py` and `feature_engineering.py`.
- **Data Exploration**: View data preview, statistics, and correlation heatmaps.
- **Ensemble Models & Visualizations**:
  - SVR/LR/RF average voting
  - XGB/LGB/LR/RF average voting
  - Weighted ensemble (optimized weights)
  - Voting Regressor (LR, RF, XGB, LGB)
  - Stacking Regressor (RF, XGB, LGB + Ridge)
  - Blending ensemble (RF, XGB, LGB + Ridge meta-model)
- **Consistent, High-Visibility Metrics**: MSE, RMSE, and R² for all ensembles, styled for clarity.
- **Rich Visualizations**: Actual vs predicted, residuals, model comparison, feature importances, learning curves, error distributions, and more.

## File Usage
- `main.py`: Main Streamlit app. Imports and uses:
  - `feature_engineering.py`: Feature engineering helpers
  - `preprocessing.py`: Preprocessing helpers
  - `utils.py`: Utility functions
  - `train.csv`: Main data source
- All other scripts and notebooks are not used by the app.

## How to Run
1. Install requirements: `pip install -r requirements.txt`
2. Run the app: `streamlit run main.py`

## Data Files
- `train.csv`: Required. Main data for training and exploration.
- `test.csv`, `sample_submission.csv`, `data_description.txt`: Not used by the app, but may be useful for reference.

## Project Structure
- Only `main.py`, `feature_engineering.py`, `preprocessing.py`, `utils.py`, and `train.csv` are required for the app.
- Other scripts and notebooks are legacy or unused.

## License
MIT License
