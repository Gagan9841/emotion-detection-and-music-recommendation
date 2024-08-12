from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib

def train_svm_model(dataset_path):
    # Load and preprocess the dataset
    data = pd.read_csv(dataset_path)
    X = data.drop('emotion', axis=1)
    y = data['emotion']
    
    # Feature Scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Evaluate the model (optional)
    accuracy = model.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    # Save the trained model
    joblib.dump(model, 'svm_model.pkl')
    
# Call the function to train and save the model
if __name__ == '__main__':
    train_svm_model('dataset/278k_song_labelled.csv')
    
