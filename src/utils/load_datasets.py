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
import src.utils as utils

# Configure logging at the module level
logger = logging.getLogger(__name__)

@dataclass
class DataLoaderConfig:
    batch_size: int = 16
    num_workers: int = 2
    shuffle: bool = False

# pylint: disable=logging-fstring-interpolation
def load_newts_dataframe(num_articles: Optional[int] = None,
                         load_test_set: Optional[bool] = False) -> pd.DataFrame:
    """
    Loads and returns the NEWTS dataset as a pandas DataFrame.
    
    :param num_articles: The number of articles to load (subset selection),
                         if None, all articles are loaded.
    :param load_test_set: Whether to load the test set, if False, the training set is loaded.
    :return: The NEWTS dataset as a pandas DataFrame.
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
            'docId': article['docId'],
            'tid1': article['tid1'],
            'tid2': article['tid2'],
        }

def load_newts_dataloader(num_articles: Optional[int] = None,
                          load_test_set: Optional[bool] = False,
                          config: Optional[DataLoaderConfig] = None) -> DataLoader:
    """
    Loads the NEWTS dataset as a DataLoader.
    
    :param num_articles: The number of articles to load (subset selection),
                         if None, all articles are loaded.
    :param load_test_set: Whether to load the test set, if False, the training set is loaded.
    :param config: An optional DataLoaderConfig object with DataLoader parameters.
                   If not provided, a default configuration is used.
    :return: A DataLoader for the NEWTS dataset.
    :raises Exception: Propagates exceptions from DataFrame loading.
    """
    if config is None:
        config = DataLoaderConfig()

    df = load_newts_dataframe(num_articles=num_articles, load_test_set=load_test_set)
    dataset = NEWTSDataset(dataframe=df)
    try:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            shuffle=config.shuffle
        )
        logger.info("Successfully created NEWTS DataLoader.")
        return dataloader
    except Exception as e:
        logger.error(f"Error setting up the DataLoader: {e}")
        raise

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