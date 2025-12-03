import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, classification_report, balanced_accuracy_score
)
from omegaconf import OmegaConf

class CancerPredictor:

    ALLOWED_TASKS = ["disease_status", "cell_type"]

    METRIC_FUNCS = {
        "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
        "f1_weighted": lambda y, p: f1_score(y, p, average="weighted"),
        "accuracy": lambda y, p: accuracy_score(y, p),
        "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, p),
        "precision_macro": lambda y, p: precision_score(y, p, average="macro"),
        "recall_macro": lambda y, p: recall_score(y, p, average="macro"),
    }

    def __init__(self, config_path):
        
        self.config = OmegaConf.load(config_path)

        self.task = self.config.task
        if self.task not in self.ALLOWED_TASKS:
            raise ValueError(f"Unknown task: {self.task}. Allowed: {self.ALLOWED_TASKS}")
        
        self.verbose = self.config.get("verbose", True)
        self.data_path = self.config.data.path

        self.random_state = self.config.training.random_state
        self.test_ratio = self.config.training.test_ratio
        self.val_ratio = self.config.training.val_ratio
        self.metrics = list(self.config.training.get("metrics", ["f1_macro"]))

        self.load_data()
    
    def load_data(self):

        self.df = pd.read_csv(self.data_path)
        task_config = self.config.data.targets[self.task]

        self.target_cols = list(task_config.cols)
        self.class_names = list(task_config.class_names)

        # get feats
        self.feature_cols = list(self.config.data.features.gene_cols)

        X = self.df[self.feature_cols].values
        y = self.df[self.target_cols].values.argmax(axis=1)  # ont-hot --> index

        # test
        self.X_temp, self.X_test, self.y_temp, self.y_test = train_test_split(
            X, y,
            test_size=self.test_ratio,
            random_state=self.random_state,
            stratify=y
        )

        # TODO: we can add more features in like: PCA, UMAP
        # self.make_features(features, remain)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_temp, self.y_temp,
            test_size=self.val_ratio,
            random_state=self.random_state,
            stratify=self.y_temp
        )

        return self


    def evaluate(self, y_true, y_pred, split_name=""):
        results = {}
        for metric_name in self.metrics:
            if metric_name not in self.METRIC_FUNCS:
                raise ValueError(f"Unknown metric: {metric_name}. Allowed: {list(self.METRIC_FUNCS.keys())}")
            score = self.METRIC_FUNCS[metric_name](y_true, y_pred)
            results[metric_name] = score
        
        print(f"\n[{split_name}] Evaluation Results:")
        for name, score in results.items():
            print(f"  {name}: {score:.4f}")

        return results


    def train_eval(self, model):
        
        model.fit(self.X_train, self.y_train)
        self.model = model

        y_val_pred = model.predict(self.X_val)
        results = self.evaluate(self.y_val, y_val_pred, split_name="Validation")
        self.val_results = results

        return model, results


    def test_eval(self):
        y_test_pred = self.model.predict(self.X_test)
        results = self.evaluate(self.y_test, y_test_pred, split_name="Test")
    
        print(f"\nClassification Report:")
        print(classification_report(self.y_test, y_test_pred, target_names=self.class_names))
        
        self.test_results = results
        return results


    def make_features(self, features: list, remain):
        """
        features: list of functions, each func(X) -> new features (n_samples, ?)
        remain: 
            - 1: keep all original features
            - []: keep none
            - [0, 2, ...]: keep specific columns
        """
        def _transform(X):
            parts = []
            
            if remain == 1:
                parts.append(X)
            elif remain:
                parts.append(X[:, remain])
            
            for func in features:
                new_feat = func(X)
                if new_feat.ndim == 1:
                    new_feat = new_feat.reshape(-1, 1)
                parts.append(new_feat)
            
            if not parts:
                return X
            
            return np.concatenate(parts, axis=1)
        
        self.X_train = _transform(self.X_train)
        self.X_val = _transform(self.X_val)
        self.X_test = _transform(self.X_test)
        
        return self

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    
    predictor = CancerPredictor("config.yml")
    print(f"Train: {predictor.X_train.shape}")
    print(f"Val: {predictor.X_val.shape}")
    print(f"Test: {predictor.X_test.shape}")

    model = RandomForestClassifier(random_state=42)
    predictor.train_eval(model)
    predictor.test_eval()

