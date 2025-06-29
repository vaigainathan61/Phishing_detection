import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)

    # Encode Labels
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(df['URL'], df['Label'], test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoder
