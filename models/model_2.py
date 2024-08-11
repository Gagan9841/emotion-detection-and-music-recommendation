from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def train_random_forest_model(dataset_path):
    # Load and preprocess the dataset
    data = pd.read_csv(dataset_path)
    X = data.drop('emotion', axis=1)
    y = data['emotion']
    
    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'random_forest_model.pkl')
    
    return model

# Call the function to train and save the model
if __name__ == '__main__':
    train_random_forest_model('dataset/278k_song_labelled.csv')
