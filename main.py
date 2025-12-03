from model import CancerPredictor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_models(predictor):
    config = predictor.config
    models_config = config.training.models
    random_state = config.training.random_state
    num_class = len(predictor.class_names)
    
    models = {
        "logistic_regression": LogisticRegression(
            **models_config.logistic_regression,
            random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            **models_config.random_forest,
            random_state=random_state
        ),
        "xgboost": XGBClassifier(
            **models_config.xgboost,
            random_state=random_state,
            num_class=num_class,
            eval_metric="mlogloss"
        ),
    }
    return models


def main():
    predictor = CancerPredictor("config.yml")
    
    print(f"Task: {predictor.task}")
    print(f"Classes: {predictor.class_names}")
    print(f"Train: {predictor.X_train.shape}")
    print(f"Val: {predictor.X_val.shape}")
    print(f"Test: {predictor.X_test.shape}")
    
    models = get_models(predictor)
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print("="*50)
        
        predictor.load_data()
        _, val_results = predictor.train_eval(model)
        test_results = predictor.test_eval()
        
        results[name] = {
            "val": val_results,
            "test": test_results
        }
    
    print(f"\n{'='*50}")
    print("Summary")
    print("="*50)
    for name, res in results.items():
        print(f"{name:25s} | Val F1: {res['val']['f1_macro']:.4f} | Test F1: {res['test']['f1_macro']:.4f}")


if __name__ == "__main__":
    main()