from house_price_predictor import HousePricePredictor

def main():
    # Initialize the predictor
    predictor = HousePricePredictor(train_path='train.csv', test_path='test.csv')
    
    # Run the complete pipeline
    results, best_params = predictor.run_pipeline()
    
    # Print results
    print("\nModel Performance Results:")
    print("-" * 50)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"MSE: {metrics['mse']:.2f}")
        print(f"R² Score: {metrics['r2']:.4f}")
    
    print("\nBest XGBoost Parameters:")
    print("-" * 50)
    for param, value in best_params.items():
        print(f"{param}: {value}")

if __name__ == "__main__":
    main() 