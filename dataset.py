# file data\train.parquet, data\test.parquet, data\validation.parquet
# metrics: accuracy, recall, f1-scrore, confusion matrix
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

def get_score(dataset: str = "train", get_predictions=None, n_samples: int = None):
    if get_predictions is None:
        raise ValueError("get_predictions function must be provided to compute scores.")
    if dataset == "train":
        # Load training data and compute score
        df = pd.read_parquet('data/train.parquet')
    elif dataset == "validation":
        # Load validation data and compute score
        df = pd.read_parquet('data/validation.parquet')
    else:
        raise ValueError("Invalid dataset specified. Choose from 'train', 'test', or 'validation'.")
    
    if n_samples:
        df = df.head(n_samples)
    
    predictions = []
    for idx, row in df.iterrows():
        prediction = get_predictions(row['code'])
        predictions.append(prediction)

    ground_truth = df['label']
    
    # Calculate metrics : accuracy, recall, f1-score, confusion matrix
    accuracy = accuracy_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    conf_matrix = confusion_matrix(ground_truth, predictions)
    return {
        "accuracy": accuracy,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }
    
def get_test_predictions(get_predictions=None, n_samples: int = None, save_csv: bool = True):
    if get_predictions is None:
        raise ValueError("get_predictions function must be provided to compute test predictions.")
    df = pd.read_parquet('data/test.parquet')
    if n_samples:
        df = df.head(n_samples)
    
    predictions = []
    for idx, row in df.iterrows():
        prediction = get_predictions(row['code'])
        predictions.append(prediction)
        
    # please save as ID, label in submission.csv
    if save_csv:
        submission_df = pd.DataFrame({
            "ID": df["ID"],
            "label": predictions
        })
        submission_df.to_csv("submission.csv", index=False)
    return predictions

if __name__ == "__main__":
    # Example usage
    def dummy_predictor(code):
        # A dummy predictor that randomly predicts 0 or 1
        import random
        return random.choice([0, 1])
    
    train_metrics = get_score(dataset="train", get_predictions=dummy_predictor, n_samples=1000)
    print("Train Metrics:", train_metrics)
    
    test_preds = get_test_predictions(get_predictions=dummy_predictor, n_samples=10, save_csv=False)
    print("Test Predictions:", test_preds)