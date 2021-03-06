from os import path
from utils import count_num_cpu_gpu

CAUSALM_DIR = path.dirname(path.realpath(__file__))  # This must be set to the path which specifies where the CausaLM project resides

NUM_CPU = count_num_cpu_gpu()[0]
NUM_GPU = 1

RANDOM_SEED = 212

BERT_PRETRAINED_MODEL = 'bert-base-cased'
MAX_POMS_SEQ_LENGTH = 32
MAX_SENTIMENT_SEQ_LENGTH = 384

MODELS_DIR = f"{CAUSALM_DIR}/Models"
EXPERIMENTS_DIR = f"{CAUSALM_DIR}/Experiments"

REVIEWS_MODELS_DIR = f"{MODELS_DIR}/Reviews"
REVIEWS_EXPERIMENTS_DIR = f"{EXPERIMENTS_DIR}/Reviews"

REVIEWS_FEATURES_EXPERIMENTS_DIR = f"{EXPERIMENTS_DIR}/Reviews_Features"
TASK = 'Proportion'

REVIEWS_FEATURES_DATASETS_DIR = f"{CAUSALM_DIR}/Reviews_Features/datasets"
REVIEWS_FEATURES_MODELS_DIR = f"{MODELS_DIR}/Reviews_Features"
REVIEWS_FEATURES_PRETRAIN_DIR = f"{MODELS_DIR}/Pretrain"
REVIEWS_FEATURES_PRETRAIN_DATA_DIR = f"{REVIEWS_FEATURES_PRETRAIN_DIR}/data"
REVIEWS_FEATURES_PRETRAIN_MLM_DIR = f"{REVIEWS_FEATURES_PRETRAIN_DIR}/IXT"
REVIEWS_FEATURES_PRETRAIN_IXT_DIR = f"{REVIEWS_FEATURES_PRETRAIN_DIR}/IXT"

# to change
REVIEWS_FEATURES = ("topic_price_positive")
REVIEWS_FEATURES_TREAT_CONTROL_MAP_FILE = f"{REVIEWS_FEATURES_DATASETS_DIR}/reviews_features_treat_control.json"

