'''
This file contains the utils for the sentiment vector.
'''
import os
import pandas as pd

def load_sentiment_representations(representation_type: str) -> pd.DataFrame:
    '''
    Load the sentiment representations.
    '''
    data_dir = os.getenv('SENTIMENT_DATA_PATH')
    if representation_type == 'sentiment_sentences':
        file_path = os.path.join(data_dir, 'sentiment_sentences')
        
    elif representation_type == 'sentiment_words':
        file_path = os.path.join(data_dir, 'sentiment_words.csv')
    else:
        raise ValueError(f"Invalid representation type: {representation_type}")
    return pd.read_csv(file_path)

def get_sentiment_representation_file_name(representation_type: str, language: str = 'en') -> str:
    '''
    Get the sentiment representation file name.
    '''
    if representation_type == 'sentiment_sentences':
        return f"{representation_type}_500_{language}.csv"
    elif representation_type == 'sentiment_words':
        return f"{representation_type}_2370_{language}.csv"
    else:
        raise ValueError(f"Invalid representation type: {representation_type}")
