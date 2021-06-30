from bert_constants import RANDOM_SEED
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from spacy.lang.tag_map import TAG_MAP
from utils import init_logger
import spacy
import re
import numpy as np
import pandas as pd

### BERT constants
WORDPIECE_PREFIX = "##"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"

### POS Tags constants
TOKEN_SEPARATOR = " "
WORD_POS_SEPARATOR = "_"
ADJ_POS_TAGS = ("ADJ", "ADV")
POS_TAGS_TUPLE = tuple(sorted(TAG_MAP.keys()))
POS_TAG_IDX_MAP = {str(tag): int(idx) for idx, tag in enumerate(POS_TAGS_TUPLE)}
ADJ_POS_TAGS_IDX = {"ADJ": 0, "ADV": 2}
NUM_POS_TAGS_LABELS = len(POS_TAGS_TUPLE)

sentiment_output_datasets = {0: 'negative', 1: 'positive'}


def clean_review(text: str) -> str:
    review_text = re.sub("\n", "", text)
    review_text = re.sub(" and quot;", '"', review_text)
    review_text = re.sub("<br />", "", review_text)
    review_text = re.sub(WORD_POS_SEPARATOR, "", review_text)
    review_text = re.sub("\s+", TOKEN_SEPARATOR, review_text)
    # review_text = re.sub(";", ",", review_text)
    return review_text.strip()


class PretrainedPOSTagger:

    """This module requires en_core_web_lg model to be installed"""
    tagger = spacy.load("en_core_web_lg")

    @staticmethod
    def tag_review(review: str) -> str:
        review_text = clean_review(review)
        tagged_review = [f"{token.text}{WORD_POS_SEPARATOR}{token.pos_}"
                         for token in PretrainedPOSTagger.tagger(review_text)]
        return TOKEN_SEPARATOR.join(tagged_review)


def split_data(df: DataFrame, path: str, prefix: str, label_column: str = "label"):
    train, test = train_test_split(df, test_size=0.2, stratify=df[label_column], random_state=RANDOM_SEED)
    train, dev = train_test_split(train, test_size=0.2, stratify=train[label_column], random_state=RANDOM_SEED)
    df.sort_index().to_csv(f"{path}/{prefix}_all.csv")
    # train.sort_index().to_csv(f"{path}/{prefix}_train.csv")
    dev.sort_index().to_csv(f"{path}/{prefix}_dev.csv")
    test.sort_index().to_csv(f"{path}/{prefix}_test.csv")
    return train, dev, test


def bias_random_sampling(df: DataFrame, biasing_factor: float, seed: int = RANDOM_SEED):
    return df.sample(frac=biasing_factor, random_state=seed)


def validate_dataset(df, stats_columns, bias_column, label_column, logger=None):
    if not logger:
        logger = init_logger("validate_dataset")
    logger.info(f"Num reviews: {len(df)}")
    logger.info(f"{df.columns}")
    for col in df.columns:
        if col.endswith("_label"):
            logger.info(f"{df[col].value_counts(dropna=False)}\n")
    for col in stats_columns:
        col_vals = df[col]
        logger.info(f"{col} statistics:")
        logger.info(f"Min: {col_vals.min()}")
        logger.info(f"Max: {col_vals.max()}")
        logger.info(f"Std: {col_vals.std()}")
        logger.info(f"Mean: {col_vals.mean()}")
        logger.info(f"Median: {col_vals.median()}")
    logger.info(f"Correlation between {bias_column} and {label_column}: "
                f"{df[bias_column].corr(df[label_column].astype(float))}\n")
