# import pandas as pd

# # Load the dataset
# dataset_path = '/var/www/html/8thproject/emotion-detection-and-music-recommendation/dataset/278k_labelled_uri.csv'  # Update with the path to your dataset
# df = pd.read_csv(dataset_path)

# # Define a function to recommend songs based on emotion
# def recommend_songs(emotion, num_recommendations=10, confidence):
#     # Map emotions to labels
#     emotion_labels = {
#         'sad': 0,
#         'happy': 1,
#         'energetic': 2,
#         'calm': 3
#     }
#     if emotion not in emotion_labels:
#         raise ValueError("Emotion not recognized. Available emotions: 'sad', 'happy', 'energetic', 'calm'")
    
#     # Filter songs based on the emotion
#     emotion_label = emotion_labels[emotion]
#     filtered_songs = df[df['labels'] == emotion_label]
    
#     # Sort by a chosen feature (e.g., energy) for recommendations
#     recommended_songs = filtered_songs.sort_values(by='energy', ascending=False).head(num_recommendations)
    
#     # Include Spotify links
#     recommended_songs['spotify_link'] = recommended_songs['uri'].apply(lambda uri: f'https://open.spotify.com/track/{uri.split(":")[2]}')
    
#     # Return the relevant columns with the Spotify link
#     return recommended_songs[['danceability', 'energy', 'valence', 'spotify_link']]

# # Example usage
# emotion = 'happy'  # Replace with detected emotion
# recommended_songs = recommend_songs(emotion)

# print("Recommended Songs:")
# print(recommended_songs)



import pandas as pd

# Load both datasets
dataset_278k_path = '/var/www/html/8thproject/emotion-detection-and-music-recommendation/dataset/278k_labelled_uri.csv'
dataset_muse_path = '/var/www/html/8thproject/emotion-detection-and-music-recommendation/dataset/muse_dataset.csv'
df_278k = pd.read_csv(dataset_278k_path)
df_muse = pd.read_csv(dataset_muse_path)

# Define a function to recommend songs based on emotion and confidence
def recommend_songs(emotion, num_recommendations=10, confidence=1.0):
    # Map emotions to labels for the 278k dataset
    emotion_labels_278k = {
        'sad': 0,
        'happy': 1,
        'energetic': 2,
        'calm': 3
    }
    
    if emotion not in emotion_labels_278k:
        raise ValueError("Emotion not recognized. Available emotions: 'sad', 'happy', 'energetic', 'calm'")
    
    # Filter songs from the 278k dataset based on the emotion
    emotion_label_278k = emotion_labels_278k[emotion]
    filtered_songs_278k = df_278k[df_278k['labels'] == emotion_label_278k]
    
    # Normalize confidence (0.0 to 1.0) and map to MuSe dataset scores
    valence_threshold = confidence * 2  # Scale confidence to match valence/arousal ranges
    arousal_threshold = confidence * 2

    # Filter songs from the MuSe dataset based on emotion
    if emotion == 'happy':
        filtered_songs_muse = df_muse[df_muse['valence_tags'] >= valence_threshold]
    elif emotion == 'sad':
        filtered_songs_muse = df_muse[df_muse['valence_tags'] <= (10 - valence_threshold)]
    elif emotion == 'energetic':
        filtered_songs_muse = df_muse[df_muse['arousal_tags'] >= arousal_threshold]
    elif emotion == 'calm':
        filtered_songs_muse = df_muse[df_muse['arousal_tags'] <= (10 - arousal_threshold)]
    
    # Sort and select top recommendations from both datasets
    recommended_songs_278k = filtered_songs_278k.sort_values(by='energy', ascending=False).head(num_recommendations)
    recommended_songs_muse = filtered_songs_muse.sort_values(by='arousal_tags', ascending=False).head(num_recommendations)

    # Combine the two recommendation sets
    combined_recommendations = pd.concat([recommended_songs_278k, recommended_songs_muse])

    # Add Spotify links
    combined_recommendations['spotify_link'] = combined_recommendations.apply(
        lambda row: f'https://open.spotify.com/track/{row["uri"].split(":")[2]}' if 'uri' in row else f'https://open.spotify.com/track/{row["spotify_id"]}', 
        axis=1
    )

    # Return the relevant columns with the Spotify link
    return combined_recommendations[['danceability', 'energy', 'valence', 'spotify_link']]

# Example usage
emotion = 'happy'  # Replace with detected emotion
confidence = 0.8  # Replace with detected confidence
recommended_songs = recommend_songs(emotion, confidence=confidence)

print("Recommended Songs:")
print(recommended_songs)

