# Auto-generated from e-commerce-fraud-detection-gb-acc-0-99.ipynb
# Exported at 2025-11-19T09:13:06.649435Z

# In[1]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class FraudDetectionModel:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results_no_oversampling = {}
        self.results_oversampling = {}
        
    def load_and_preprocess_data(self):
        print("Loading and preprocessing data...")
        self.df = pd.read_csv(self.file_path)
        
        # Convert transaction_time to datetime and extract features
        self.df['transaction_time'] = pd.to_datetime(self.df['transaction_time'])
        self.df['transaction_hour'] = self.df['transaction_time'].dt.hour
        self.df['transaction_day'] = self.df['transaction_time'].dt.dayofweek
        self.df['transaction_month'] = self.df['transaction_time'].dt.month
        self.df['is_weekend'] = (self.df['transaction_time'].dt.dayofweek >= 5).astype(int)
        
        # Create country mismatch feature
        self.df['country_mismatch'] = (self.df['country'] != self.df['bin_country']).astype(int)
        
        # Create transaction amount ratios
        self.df['amount_vs_avg'] = self.df['amount'] / (self.df['avg_amount_user'] + 1e-8)
        
        # Create time since first transaction
        self.df['days_since_first'] = self.df.groupby('user_id')['transaction_time'].transform(
            lambda x: (x - x.min()).dt.total_seconds() / (24 * 3600)
        )
        
        # Drop unnecessary columns
        columns_to_drop = ['transaction_id', 'transaction_time']
        self.df = self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns])
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Fraud rate: {self.df['is_fraud'].mean():.4f} ({self.df['is_fraud'].mean():.2%})")
        
        return self.df
    
    def prepare_data(self):
        """Prepare features with simple encoding"""
        # Separate features and target
        y = self.df['is_fraud']
        X = self.df.drop('is_fraud', axis=1)
        
        # Simple one-hot encoding for categorical variables
        categorical_cols = ['country', 'bin_country', 'channel', 'merchant_category', 
                           'avs_match', 'cvv_result', 'three_ds_flag']
        
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['account_age_days', 'total_transactions_user', 'avg_amount_user', 
                         'amount', 'shipping_distance_km', 'transaction_hour', 'transaction_day',
                         'transaction_month', 'amount_vs_avg', 'days_since_first']
        
        # Only use columns that exist
        numerical_cols = [col for col in numerical_cols if col in self.X_train.columns]
        
        self.X_train[numerical_cols] = scaler.fit_transform(self.X_train[numerical_cols])
        self.X_test[numerical_cols] = scaler.transform(self.X_test[numerical_cols])
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Training fraud rate: {self.y_train.mean():.4f}")
        print(f"Test fraud rate: {self.y_test.mean():.4f}")
        print(f"Final feature count: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def manual_oversample_minority(self, X, y, target_ratio=0.3):
        """Simple manual oversampling"""
        # Separate majority and minority classes
        majority_class = X[y == 0]
        minority_class = X[y == 1]
        
        # Calculate how many samples to generate
        n_majority = len(majority_class)
        n_minority = len(minority_class)
        desired_minority = int(n_majority * target_ratio)
        n_to_generate = desired_minority - n_minority
        
        if n_to_generate > 0:
            # Simple replication
            indices = np.random.choice(len(minority_class), n_to_generate, replace=True)
            synthetic_samples = minority_class.iloc[indices].copy()
            
            # Combine original and synthetic
            X_balanced = pd.concat([X, synthetic_samples], axis=0)
            y_balanced = pd.concat([y, pd.Series([1] * n_to_generate)], axis=0)
            
            # Shuffle the data
            shuffle_idx = np.random.permutation(len(X_balanced))
            X_balanced = X_balanced.iloc[shuffle_idx].reset_index(drop=True)
            y_balanced = y_balanced.iloc[shuffle_idx].reset_index(drop=True)
            
            print(f"After manual oversampling: {np.bincount(y_balanced)}")
            return X_balanced, y_balanced
        
        return X, y
    
    def initialize_models(self):
        """Initialize models with imbalance handling"""
        # Calculate scale_pos_weight for 2% fraud rate
        scale_pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        
        self.models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                class_weight='balanced',
                C=0.1,
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        }
        
        print(f"Scale pos weight calculated: {scale_pos_weight:.2f}")
        return self.models
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        auc_pr = average_precision_score(y_test, y_pred_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'model': model_name,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'accuracy': report['accuracy'],
            'confusion_matrix': cm
        }
        
        return results, y_pred_proba
    
    def train_single_approach(self, use_oversampling=False):
        """Train models with specified approach"""
        print("Starting model training and evaluation...")
        
        # Handle imbalance with manual oversampling if requested
        if use_oversampling:
            print("Applying manual oversampling...")
            X_train_processed, y_train_processed = self.manual_oversample_minority(
                self.X_train, self.y_train, target_ratio=0.3
            )
        else:
            X_train_processed, y_train_processed = self.X_train, self.y_train
        
        # Train and evaluate models
        results_approach = {}
        
        for model_name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name}...")
            print(f"{'='*50}")
            
            try:
                # Train model
                model.fit(X_train_processed, y_train_processed)
                
                # Evaluate
                results, y_proba = self.evaluate_model(model, self.X_test, self.y_test, model_name)
                
                results_approach[model_name] = (results, y_proba)
                
                # Print results
                print(f"\n{model_name} Results:")
                print(f"AUC-ROC: {results['auc_roc']:.4f}")
                print(f"AUC-PR: {results['auc_pr']:.4f}")
                print(f"Precision: {results['precision']:.4f}")
                print(f"Recall: {results['recall']:.4f}")
                print(f"F1-Score: {results['f1_score']:.4f}")
                print(f"Accuracy: {results['accuracy']:.4f}")
                print(f"Confusion Matrix:")
                print(results['confusion_matrix'])
                
            except Exception as e:
                print(f"Error training {model_name}: {e}")
                continue
        
        return results_approach

    def compare_approaches(self):
        """Compare both approaches and show final results"""
        print("\n" + "="*80)
        print("COMPARISON OF APPROACHES")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        
        for model_name in self.models.keys():
            if model_name in self.results_no_oversampling and model_name in self.results_oversampling:
                results_no = self.results_no_oversampling[model_name][0]
                results_over = self.results_oversampling[model_name][0]
                
                comparison_data.append({
                    'Model': model_name,
                    'Approach': 'Class Weights',
                    'AUC-ROC': results_no['auc_roc'],
                    'AUC-PR': results_no['auc_pr'],
                    'Precision': results_no['precision'],
                    'Recall': results_no['recall'],
                    'F1-Score': results_no['f1_score'],
                    'Accuracy': results_no['accuracy']
                })
                
                comparison_data.append({
                    'Model': model_name,
                    'Approach': 'Oversampling',
                    'AUC-ROC': results_over['auc_roc'],
                    'AUC-PR': results_over['auc_pr'],
                    'Precision': results_over['precision'],
                    'Recall': results_over['recall'],
                    'F1-Score': results_over['f1_score'],
                    'Accuracy': results_over['accuracy']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Display comparison
        print("\nDetailed Comparison:")
        print("-" * 120)
        for model_name in self.models.keys():
            if model_name in self.results_no_oversampling:
                model_data = comparison_df[comparison_df['Model'] == model_name]
                print(f"\n{model_name}:")
                print(model_data.to_string(index=False, float_format='%.4f'))
                print("-" * 120)
        
        # Find best model
        best_model = None
        best_auc_pr = 0
        best_approach = ""
        
        for model_name in self.models.keys():
            if model_name in self.results_no_oversampling:
                auc_pr_no = self.results_no_oversampling[model_name][0]['auc_pr']
                if auc_pr_no > best_auc_pr:
                    best_auc_pr = auc_pr_no
                    best_model = model_name
                    best_approach = "Class Weights"
            
            if model_name in self.results_oversampling:
                auc_pr_over = self.results_oversampling[model_name][0]['auc_pr']
                if auc_pr_over > best_auc_pr:
                    best_auc_pr = auc_pr_over
                    best_model = model_name
                    best_approach = "Oversampling"
        
        # Final summary
        print(f"\n{'='*60}")
        print("FINAL SUMMARY")
        print(f"{'='*60}")
        if best_model:
            if best_approach == "Class Weights":
                best_results = self.results_no_oversampling[best_model][0]
            else:
                best_results = self.results_oversampling[best_model][0]
            
            print(f"🎯 Best Overall Model: {best_model} ({best_approach})")
            print(f"📊 Best AUC-PR: {best_results['auc_pr']:.4f}")
            print(f"📈 Best AUC-ROC: {best_results['auc_roc']:.4f}")
            print(f"⚖️  Best F1-Score: {best_results['f1_score']:.4f}")
            print(f"🔍 Recall: {best_results['recall']:.4f}")
            print(f"🎯 Precision: {best_results['precision']:.4f}")
            print(f"✅ Accuracy: {best_results['accuracy']:.4f}")
            
            # Additional insights
            print(f"\n💡 Performance Insights:")
            if best_results['recall'] > 0.8 and best_results['precision'] > 0.7:
                print("  - Excellent balance between catching fraud and minimizing false positives")
            elif best_results['recall'] > 0.8:
                print("  - Good at catching fraud, but may have many false positives")
            elif best_results['precision'] > 0.8:
                print("  - Good at minimizing false positives, but may miss some fraud")
            
            # Business impact analysis
            cm = best_results['confusion_matrix']
            print(f"\n📊 Business Impact Analysis:")
            print(f"  - Correctly detected fraud: {cm[1,1]} transactions")
            print(f"  - Missed fraud: {cm[1,0]} transactions")
            print(f"  - False alarms: {cm[0,1]} transactions")
            print(f"  - Correctly approved: {cm[0,0]} transactions")
        
        return comparison_df

# Main execution
if __name__ == "__main__":
    # Initialize and run the model
    fraud_model = FraudDetectionModel("/kaggle/input/e-commerce-fraud-detection-dataset/transactions.csv")
    
    # Load data
    fraud_model.load_and_preprocess_data()
    
    # Prepare features (simple encoding + scaling)
    fraud_model.prepare_data()
    
    # Initialize models
    fraud_model.initialize_models()
    
    # Train with class weights
    print("\n" + "="*80)
    print("1. TRAINING WITH CLASS WEIGHTS")
    print("="*80)
    fraud_model.results_no_oversampling = fraud_model.train_single_approach(use_oversampling=False)
    
    # Train with oversampling
    print("\n" + "="*80)
    print("2. TRAINING WITH OVERSAMPLING")
    print("="*80)
    fraud_model.results_oversampling = fraud_model.train_single_approach(use_oversampling=True)
    
    # Compare results
    fraud_model.compare_approaches()

