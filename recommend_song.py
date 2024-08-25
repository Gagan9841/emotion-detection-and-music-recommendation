import pandas as pd

# Load the dataset
def load_data(file_path='/var/www/html/8thproject/emotion-detection-and-music-recommendation/dataset/278k_labelled_uri.csv'):
    df = pd.read_csv(file_path)
    emotion_map = {0: 'sad', 1: 'happy', 2: 'energetic', 3: 'calm'}
    df['emotion'] = df['labels'].map(emotion_map)
    return df

# Recommend songs based on emotion and range
def recommend_songs(emotion, df, num_recommendations=10):
    print(f"Received emotion: {emotion}")
    print(f"Available emotions: {df['emotion'].unique()}")
    
    if emotion not in df['emotion'].unique():
        return {'error': 'Invalid emotion'}
    
    # Filter songs by the detected emotion
    filtered_songs = df[df['emotion'] == emotion]
    
    # Select a random sample of songs
    if len(filtered_songs) < num_recommendations:
        num_recommendations = len(filtered_songs)
    
    recommended_songs = filtered_songs.sample(n=num_recommendations)
    
    # Convert the result to a list of dictionaries
    songs = recommended_songs[['duration (ms)', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']].to_dict(orient='records')
    
    return {'songs': songs}
