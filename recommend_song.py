import pandas as pd

# Load the dataset
dataset_path = '/var/www/html/8thproject/emotion-detection-and-music-recommendation/dataset/278k_labelled_uri.csv'  # Update with the path to your dataset
df = pd.read_csv(dataset_path)

# Define a function to recommend songs based on emotion
def recommend_songs(emotion, num_recommendations=10):
    # Map emotions to labels
    emotion_labels = {
        'sad': 0,
        'happy': 1,
        'energetic': 2,
        'calm': 3
    }
    if emotion not in emotion_labels:
        raise ValueError("Emotion not recognized. Available emotions: 'sad', 'happy', 'energetic', 'calm'")
    
    # Filter songs based on the emotion
    emotion_label = emotion_labels[emotion]
    filtered_songs = df[df['labels'] == emotion_label]
    
    # Sort by a chosen feature (e.g., energy) for recommendations
    recommended_songs = filtered_songs.sort_values(by='energy', ascending=False).head(num_recommendations)
    
    # Include Spotify links
    recommended_songs['spotify_link'] = recommended_songs['uri'].apply(lambda uri: f'https://open.spotify.com/track/{uri.split(":")[2]}')
    
    # Return the relevant columns with the Spotify link
    return recommended_songs[['danceability', 'energy', 'valence', 'spotify_link']]

# Example usage
emotion = 'happy'  # Replace with detected emotion
recommended_songs = recommend_songs(emotion)

print("Recommended Songs:")
print(recommended_songs)
