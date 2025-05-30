import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from house_price_predictor import HousePricePredictor
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Cache the predictor results
@st.cache_resource
def get_predictor():
    predictor = HousePricePredictor()
    results, best_params = predictor.run_pipeline()
    return predictor, results, best_params

def create_correlation_heatmap(numerical_v2):
    """Create correlation heatmap"""
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
    return fig

def create_model_comparison_plot(results):
    """Create model comparison bar plot"""
    fig = go.Figure()
    for model_name, metrics in results.items():
        fig.add_trace(go.Bar(
            name=model_name,
            x=['MSE', 'R²'],
            y=[metrics['mse'], metrics['r2']],
            text=[f"{metrics['mse']:.2f}", f"{metrics['r2']:.4f}"]
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        barmode='group',
        template="plotly_white"
    )
    return fig

def create_residual_plot(y_test, y_pred, title="Residual Plot"):
    """Create residual plot"""
    residuals = y_test - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color='orange', opacity=0.6)
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Residuals",
        template="plotly_white"
    )
    return fig

def create_feature_importance_plot(model, feature_names, title):
    """Create feature importance plot"""
    importance = model.feature_importances_
    fig = go.Figure([go.Bar(
        x=feature_names,
        y=importance,
        marker_color='dodgerblue'
    )])
    fig.update_layout(
        title=title,
        xaxis_title="Feature",
        yaxis_title="Importance",
        template="plotly_white"
    )
    return fig

def create_learning_curves(model_name, train_errors, test_errors, title):
    """Create learning curves plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=train_errors,
        name=f'{model_name} Train',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        y=test_errors,
        name=f'{model_name} Test',
        line=dict(color='red')
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title="RMSE",
        template="plotly_white"
    )
    return fig

def create_prediction_correlation_plot(preds_df, title):
    """Create correlation plot between model predictions"""
    fig = px.imshow(
        preds_df.corr(),
        text_auto=True,
        color_continuous_scale='Viridis',
        title=title
    )
    return fig

def create_error_improvement_plot(best_base_error, blended_error, title):
    """Create error improvement plot"""
    improvement = best_base_error - blended_error
    fig = px.histogram(
        improvement,
        nbins=30,
        title=title,
        labels={'value': 'Improvement in Absolute Error'}
    )
    return fig

def create_ensemble_comparison_plot(simple_avg_results, weighted_avg_results):
    """Create comparison plot between simple and weighted ensemble"""
    compare_df = pd.DataFrame({
        'Model': ['Simple Avg', 'Weighted Avg'],
        'MSE': [simple_avg_results['mse'], weighted_avg_results['mse']],
        'R2': [simple_avg_results['r2'], weighted_avg_results['r2']]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=compare_df['Model'],
        y=compare_df['MSE'],
        name='MSE',
        marker_color='lightsalmon'
    ))
    fig.add_trace(go.Bar(
        x=compare_df['Model'],
        y=compare_df['R2'],
        name='R² Score',
        marker_color='lightblue',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Simple vs Weighted Ensemble Performance",
        xaxis_title="Model",
        yaxis=dict(title='MSE', side='left'),
        yaxis2=dict(title='R² Score', overlaying='y', side='right'),
        barmode='group',
        template='plotly_white'
    )
    return fig

def create_parallel_coordinate_plot(opt_history):
    """Create parallel coordinate plot for hyperparameter optimization"""
    if 'params' not in opt_history:
        return None
        
    # Create a DataFrame with all parameters
    params_df = pd.DataFrame(opt_history['params'])
    params_df['MSE'] = opt_history['MSE']
    
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=params_df['MSE'],
                colorscale='Viridis',
                showscale=True
            ),
            dimensions=[dict(range=[params_df[col].min(), params_df[col].max()],
                           label=col,
                           values=params_df[col]) for col in params_df.columns]
        )
    )
    
    fig.update_layout(
        title='Parallel Coordinate Plot of Hyperparameter Optimization',
        template='plotly_white'
    )
    return fig

def create_contour_plot(opt_history):
    """Create contour plot for hyperparameter interactions"""
    if 'params' not in opt_history:
        return None
        
    # Get the two most important parameters
    params_df = pd.DataFrame(opt_history['params'])
    if 'param_importances' in opt_history:
        important_params = opt_history['param_importances'].nlargest(2, 'importance')['parameter'].tolist()
        if len(important_params) >= 2:
            param1, param2 = important_params[:2]
            
            fig = go.Figure(data=
                go.Contour(
                    x=params_df[param1],
                    y=params_df[param2],
                    z=opt_history['MSE'],
                    colorscale='Viridis',
                    showscale=True
                )
            )
            
            fig.update_layout(
                title=f'Contour Plot: {param1} vs {param2}',
                xaxis_title=param1,
                yaxis_title=param2,
                template='plotly_white'
            )
            return fig
    return None

def main():
    st.set_page_config(page_title="House Price Prediction Analysis", layout="wide")
    st.title("House Price Prediction Analysis")
    
    # Initialize predictor with caching
    with st.spinner("Loading and processing data (this may take a few minutes)..."):
        predictor, results, best_params = get_predictor()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio(
        "Select Section",
        ["Overview", "Base Models", "Voting Ensemble", "Stacking Ensemble", "Blending", "Hyperparameter Optimization", "Ensemble Comparison"]
    )
    
    if section == "Overview":
        st.header("Overview")
        st.subheader("Correlation Analysis")
        st.plotly_chart(create_correlation_heatmap(predictor.numerical), use_container_width=True)
        
        st.subheader("Model Performance Comparison")
        st.plotly_chart(create_model_comparison_plot(results), use_container_width=True)
    
    elif section == "Base Models":
        st.header("Base Models Analysis")
        
        # Linear Regression
        st.subheader("Linear Regression")
        lr_pred = predictor.models['lr'].predict(predictor.final_test)
        st.plotly_chart(create_residual_plot(predictor.y_test, lr_pred, "Linear Regression Residuals"), use_container_width=True)
        
        # Random Forest
        st.subheader("Random Forest")
        rf_pred = predictor.models['rf'].predict(predictor.final_test)
        st.plotly_chart(create_residual_plot(predictor.y_test, rf_pred, "Random Forest Residuals"), use_container_width=True)
        st.plotly_chart(create_feature_importance_plot(
            predictor.models['rf'],
            predictor.final_train.columns,
            "Random Forest Feature Importance"
        ), use_container_width=True)
        
        # XGBoost
        st.subheader("XGBoost")
        xgb_pred = predictor.models['xgb'].predict(predictor.final_test)
        st.plotly_chart(create_residual_plot(predictor.y_test, xgb_pred, "XGBoost Residuals"), use_container_width=True)
        st.plotly_chart(create_feature_importance_plot(
            predictor.models['xgb'],
            predictor.final_train.columns,
            "XGBoost Feature Importance"
        ), use_container_width=True)
        
        # XGBoost Learning Curves
        if hasattr(predictor.models['xgb'], 'evals_result_'):
            xgb_eval = predictor.models['xgb'].evals_result_
            st.plotly_chart(create_learning_curves(
                'XGBoost',
                xgb_eval['validation_0']['rmse'],
                xgb_eval['validation_1']['rmse'],
                "XGBoost Learning Curves"
            ), use_container_width=True)
        
        # LightGBM
        st.subheader("LightGBM")
        lgb_pred = predictor.models['lgb'].predict(predictor.final_test)
        st.plotly_chart(create_residual_plot(predictor.y_test, lgb_pred, "LightGBM Residuals"), use_container_width=True)
        st.plotly_chart(create_feature_importance_plot(
            predictor.models['lgb'],
            predictor.final_train.columns,
            "LightGBM Feature Importance"
        ), use_container_width=True)
        
        # LightGBM Learning Curves
        if hasattr(predictor.models['lgb'], 'evals_result_'):
            lgb_eval = predictor.models['lgb'].evals_result_
            st.plotly_chart(create_learning_curves(
                'LightGBM',
                lgb_eval['training']['rmse'],
                lgb_eval['valid_1']['rmse'],
                "LightGBM Learning Curves"
            ), use_container_width=True)
    
    elif section == "Voting Ensemble":
        st.header("Voting Ensemble Analysis")
        voting_pred = predictor.voting_reg.predict(predictor.final_test)
        
        st.subheader("Performance Metrics")
        st.write(f"MSE: {results['voting']['mse']:.2f}")
        st.write(f"R² Score: {results['voting']['r2']:.4f}")
        
        st.subheader("Residual Plot")
        st.plotly_chart(create_residual_plot(predictor.y_test, voting_pred, "Voting Ensemble Residuals"), use_container_width=True)
        
        st.subheader("Prediction vs Actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictor.y_test,
            y=voting_pred,
            mode='markers',
            name='Predicted vs Actual',
            marker=dict(color='blue', opacity=0.6)
        ))
        fig.add_trace(go.Scatter(
            x=[predictor.y_test.min(), predictor.y_test.max()],
            y=[predictor.y_test.min(), predictor.y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        fig.update_layout(
            title="Actual vs Predicted (Voting Ensemble)",
            xaxis_title="Actual",
            yaxis_title="Predicted",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif section == "Stacking Ensemble":
        st.header("Stacking Ensemble Analysis")
        stacking_pred = predictor.stacking_reg.predict(predictor.final_test)
        
        st.subheader("Performance Metrics")
        st.write(f"MSE: {results['stacking']['mse']:.2f}")
        st.write(f"R² Score: {results['stacking']['r2']:.4f}")
        
        st.subheader("Residual Plot")
        st.plotly_chart(create_residual_plot(predictor.y_test, stacking_pred, "Stacking Ensemble Residuals"), use_container_width=True)
        
        st.subheader("Residual Distribution")
        residuals = predictor.y_test - stacking_pred
        fig = px.histogram(
            residuals,
            nbins=30,
            marginal='box',
            title="Residual Distribution (Stacking Ensemble)",
            labels={'value': 'Prediction Error'}
        )
        fig.update_layout(xaxis_title="Residual", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
        
        # Add meta-model coefficients plot
        if hasattr(predictor.stacking_reg, 'final_estimator_'):
            coef_df = pd.DataFrame({
                'Model': ['RandomForest', 'XGBoost', 'LightGBM'],
                'Coefficient': predictor.stacking_reg.final_estimator_.coef_
            })
            fig = px.bar(
                coef_df,
                x='Model',
                y='Coefficient',
                title='Meta-Model Coefficients (Stacking Weights)',
                text='Coefficient'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif section == "Blending":
        st.header("Blending Analysis")
        
        # Get predictions from base models
        rf_pred = predictor.models['rf'].predict(predictor.final_test)
        xgb_pred = predictor.models['xgb'].predict(predictor.final_test)
        lgb_pred = predictor.models['lgb'].predict(predictor.final_test)
        
        # Create correlation matrix of predictions
        preds_df = pd.DataFrame({
            'RandomForest': rf_pred,
            'XGBoost': xgb_pred,
            'LightGBM': lgb_pred
        })
        
        st.subheader("Correlation Between Base Model Predictions")
        st.plotly_chart(create_prediction_correlation_plot(
            preds_df,
            'Correlation Matrix of Base Model Predictions'
        ), use_container_width=True)
        
        st.subheader("Prediction Errors Distribution")
        errors_df = pd.DataFrame({
            'RandomForest Error': np.abs(predictor.y_test - rf_pred),
            'XGBoost Error': np.abs(predictor.y_test - xgb_pred),
            'LightGBM Error': np.abs(predictor.y_test - lgb_pred)
        })
        
        fig = go.Figure()
        for col in errors_df.columns:
            fig.add_trace(go.Box(y=errors_df[col], name=col))
        fig.update_layout(
            title='Prediction Errors Distribution by Model',
            yaxis_title='Absolute Error',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add error improvement plot
        best_base_error = np.minimum.reduce([
            np.abs(predictor.y_test - rf_pred),
            np.abs(predictor.y_test - xgb_pred),
            np.abs(predictor.y_test - lgb_pred)
        ])
        
        # Get blended predictions
        blended_pred = predictor.blending_reg.predict(predictor.final_test)
        blended_error = np.abs(predictor.y_test - blended_pred)
        
        st.plotly_chart(create_error_improvement_plot(
            best_base_error,
            blended_error,
            'Error Improvement by Blending over Best Base Model'
        ), use_container_width=True)
        
        # Add scatter plot of predictions
        fig = px.scatter(
            x=rf_pred,
            y=xgb_pred,
            labels={'x': 'RandomForest Predictions', 'y': 'XGBoost Predictions'},
            title='Scatter of RF vs XGB Predictions'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif section == "Hyperparameter Optimization":
        st.header("Hyperparameter Optimization Results")
        
        st.subheader("Best Parameters")
        for param, value in best_params.items():
            st.write(f"{param}: {value}")
        
        st.subheader("Optimization History")
        opt_history = predictor.get_optimization_history()
        if opt_history is not None:
            # Optimization progress plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=opt_history['Trial'],
                y=opt_history['MSE'],
                mode='lines+markers',
                name='Validation MSE'
            ))
            fig.update_layout(
                title='Optimization Progress',
                xaxis_title='Trial Number',
                yaxis_title='Validation MSE',
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Hyperparameter importance plot
            if 'param_importances' in opt_history:
                fig = px.bar(
                    opt_history['param_importances'],
                    x='parameter',
                    y='importance',
                    title='Hyperparameter Importance'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Parallel coordinate plot
            parallel_coord_fig = create_parallel_coordinate_plot(opt_history)
            if parallel_coord_fig is not None:
                st.subheader("Parameter Interactions")
                st.plotly_chart(parallel_coord_fig, use_container_width=True)
            
            # Contour plot
            contour_fig = create_contour_plot(opt_history)
            if contour_fig is not None:
                st.subheader("Parameter Contour Plot")
                st.plotly_chart(contour_fig, use_container_width=True)
        
        st.subheader("Feature Importance from Optimized Model")
        optimized_xgb = XGBRegressor(**best_params)
        optimized_xgb.fit(predictor.final_train, predictor.y_train)
        
        st.plotly_chart(create_feature_importance_plot(
            optimized_xgb,
            predictor.final_train.columns,
            "Optimized XGBoost Feature Importance"
        ), use_container_width=True)

    elif section == "Ensemble Comparison":
        st.header("Ensemble Methods Comparison")
        
        # Get predictions from both ensemble methods
        simple_avg_pred = predictor.voting_reg.predict(predictor.final_test)
        weighted_avg_pred = predictor.stacking_reg.predict(predictor.final_test)
        
        # Calculate metrics
        simple_avg_results = {
            'mse': mean_squared_error(predictor.y_test, simple_avg_pred),
            'r2': r2_score(predictor.y_test, simple_avg_pred)
        }
        
        weighted_avg_results = {
            'mse': mean_squared_error(predictor.y_test, weighted_avg_pred),
            'r2': r2_score(predictor.y_test, weighted_avg_pred)
        }
        
        # Create comparison plot
        st.plotly_chart(create_ensemble_comparison_plot(
            simple_avg_results,
            weighted_avg_results
        ), use_container_width=True)
        
        # Residual distribution for weighted ensemble
        residuals = predictor.y_test - weighted_avg_pred
        fig = px.histogram(
            residuals,
            nbins=30,
            marginal='box',
            title="Residual Distribution (Weighted Ensemble)",
            labels={'value': 'Prediction Error'}
        )
        fig.update_layout(xaxis_title="Residual", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 