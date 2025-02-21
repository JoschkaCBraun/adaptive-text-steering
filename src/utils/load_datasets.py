"""
load_datasets.py

Utility functions for different datasets including the NEWTS dataset.
"""
# Standard library imports
import os
import logging
from typing import Dict, Optional

# Third-party imports
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

# Local imports
from config.experiment_config import ExperimentConfig
import src.utils as utils

# Configure logging at the module level
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    batch_size: int = 16
    num_workers: int = 2
    shuffle: bool = False

# pylint: disable=logging-fstring-interpolation
def load_newts_dataframe(num_articles: Optional[int] = None, load_test_set: bool = False) -> pd.DataFrame:
    """
    Loads and returns the NEWTS dataset as a pandas DataFrame.
    
    :param num_articles: The number of articles to load (subset selection).
    :param load_test_set: Whether to load the test set.
    :return: The NEWTS dataset as a DataFrame.
    :raises ValueError: If the number of articles exceeds allowed limits.
    :raises FileNotFoundError: If the dataset file does not exist.
    """
    datasets_path = utils.get_path("DATASETS_PATH")

    if load_test_set:
        if num_articles is None:
            num_articles = 600
        if num_articles > 600:
            raise ValueError("When loading the test set, num_articles must be 600 or fewer")
        dataset_name = 'NEWTS_test_600'
    else:
        if num_articles is None:
            num_articles = 2400
        if num_articles > 2400:
            raise ValueError("When loading the training set, num_articles must be 2400 or fewer")
        dataset_name = 'NEWTS_train_2400'
    
    file_path = os.path.join(datasets_path, "NEWTS", f"{dataset_name}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        df.rename(columns={df.columns[0]: 'article_idx'}, inplace=True)
        logger.info(f"Successfully loaded {dataset_name} dataset.")
        return df
    except Exception as e:
        logger.error(f"Error reading {dataset_name} dataset: {e}")
        raise

class NEWTSDataset(Dataset):
    """
    Dataset class for the NEWTS dataset.
    """
    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.data = dataframe

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        article = self.data.iloc[idx]
        return {
            'article': article['article'],
            'summary1': article['summary1'],
            'summary2': article['summary2'],
            'article_idx': article['article_idx'],
            'tid1': article['tid1'],
            'tid2': article['tid2'],
        }

def get_dataloader(dataset_name: str, config: ExperimentConfig) -> DataLoader:
    """
    Sets up and returns a DataLoader for either the NEWTS training or testing set.

    :param dataset_name: The name of the dataset to set up the DataLoader for.
    :return: A DataLoader for the specified NEWTS dataset.
    """
    try:
        dataloader = DataLoader(dataset=NEWTSDataset(dataframe=load_dataset(dataset_name)),
                                batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS,
                                shuffle=config.SHUFFLE,)
        return dataloader
    except Exception as e:
        logging.error(f"Error setting up the DataLoader: {e}")
        return None

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """
    Loads and returns the specified dataset as a Pandas DataFrame.
    
    :param dataset_name: The name of the dataset to load.
    :return: The specified dataset DataFrame or None if an error occurs.
    """
    datasets_path = utils.get_path("DATASETS_PATH")
    if dataset_name in ['imdb_sentiment_train', 'imdb_sentiment_test']:
        file_path = os.path.join(datasets_path, "imdb_sentiment", f"{dataset_name}.parquet")
    elif dataset_name in ['simplicity_wiki_doc_train', 'simplicity_wiki_doc_test']:
        file_path = os.path.join(datasets_path, "simplicity_wiki_doc", f"{dataset_name}.parquet")
    else:
        logging.error(f"Given dataset_name {dataset_name} is not valid")
        return None

    try:
        if dataset_name in ['imdb_sentiment_train', 'imdb_sentiment_test']:
            df = pd.read_parquet(file_path)

        logging.info(f"Successfully loaded {dataset_name} dataset.")
        return df
    except Exception as e:
        logging.error(f"Error reading {dataset_name} dataset: {e}")
        return None