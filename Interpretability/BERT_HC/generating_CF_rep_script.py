import pandas as pd
import json
import os
from Reviews_Features.lm_finetune import mlm_finetune_on_pregenrated, pregenerate_training_data,\
    features_finetune_on_pregenerated
from bert_constants import REVIEWS_FEATURES_DATASETS_DIR
from Predicting_Decisions_in_Language_Based_Persuasion_Games import create_bert_embedding
from Predicting_Decisions_in_Language_Based_Persuasion_Games import create_seq_models_input
import sys


def main(is_mlm: bool):
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    path = f"{REVIEWS_FEATURES_DATASETS_DIR}/experiment_manage.csv"
    experiment_manage_df = pd.read_csv(path)
    for index, row in experiment_manage_df.iterrows():
        pregenerate_training_data.main(row.item())
        if not is_mlm:
            features_finetune_on_pregenerated.main(row.item())
        else:
            mlm_finetune_on_pregenrated.main(row.item())
        create_bert_embedding.main([row.item()])
        create_seq_models_input.main([f'bert_embedding_for_feature_{row.item()}'])


if __name__ == '__main__':
    if len(sys.argv) > 1:
        is_mlm = sys.argv[1]  # whether to run mlm or features predictions
    else:
        is_mlm = False
    if is_mlm == 'False':
        is_mlm = False
    main(is_mlm)
