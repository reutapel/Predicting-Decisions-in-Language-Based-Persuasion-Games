from typing import *
import pandas as pd
import numpy as np
import tqdm
from overrides import overrides
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, ListField, ArrayField, SequenceLabelField
from float_label_field import FloatLabelField
from allennlp.data.instance import Instance
import joblib
import copy


class LSTMDatasetReader(DatasetReader):
    """
    DatasetReader for LSTM models that predict for each round the DM's decision
    """
    def __init__(self,
                 lazy: bool = False,
                 label_column: Optional[str] = 'labels',
                 pair_ids: list = None,
                 num_attention_heads: int=8,
                 use_transformer: bool=False,
                 use_raisha_attention: bool=False,
                 use_raisha_LSTM: bool=False,
                 raisha_num_features: int=0) -> None:
        """

        :param lazy:
        :param label_column:
        :param pair_ids:
        :param num_attention_heads:
        :param use_transformer: if the model is a transformer
        :param use_raisha_attention: if the model first use attention to create a raisha vector
        :param use_raisha_LSTM: add the raisha rounds as is to the LSTM --> need to put -1 in the saifa features to
        get the same vector shape in all rounds
        """
        super().__init__(lazy)
        self._label_column = label_column
        self.num_features = 0
        self.raisha_num_features = raisha_num_features
        self.num_labels = 2
        self.pair_ids = pair_ids
        self.input_dim = 0
        self.use_transformer = use_transformer
        self.num_attention_heads = num_attention_heads
        self.use_raisha_attention = use_raisha_attention
        self.use_raisha_LSTM = use_raisha_LSTM

        if self.use_raisha_LSTM and self.raisha_num_features == 0:
            print(f'When using use_raisha_LSTM, raisha_num_features must be the correct number and not 0')
            raise TypeError

    @overrides
    def text_to_instance(self, features_list: List[ArrayField], raisha_text_list: List[ArrayField],
                         labels: List[str] = None, metadata: Dict=None) -> Instance:
        sentence_field = ListField(features_list)
        fields = {'sequence_review': sentence_field}

        if raisha_text_list is not None:
            fields['raisha_sequence_review'] = ListField(raisha_text_list)

        if labels is not None:
            sentence_field_for_labels = copy.deepcopy(sentence_field)
            if self.use_raisha_LSTM:
                sentence_field_for_labels.field_list = sentence_field_for_labels.field_list[metadata['raisha']:]
            seq_labels_field = SequenceLabelField(labels=labels, sequence_field=sentence_field_for_labels)
            fields['seq_labels'] = seq_labels_field
            reg_labels = [0 if label == 'hotel' else 1 for label in labels]
            reg_label_field = FloatLabelField(sum(reg_labels) / len(reg_labels))
            fields['reg_labels'] = reg_label_field

        if metadata is not None:
            fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        This function takes a filename, read the data and produces a stream of Instances
        :param str file_path: the path to the file with the data
        :return:
        """
        # Load the data
        if 'csv' in file_path:
            df = pd.read_csv(file_path)
        else:
            df = joblib.load(file_path)

        # if we run with CV we need the pair ids to use
        if self.pair_ids is not None:
            df = df.loc[df.pair_id.isin(self.pair_ids)]

        # get the reviews and label columns -> no metadata, and metadata columns
        metadata_columns = ['raisha', 'pair_id', 'sample_id']
        rounds = list(range(1, 11))  # rounds 1-10

        for i, row in tqdm.tqdm(df.iterrows()):
            text_list = list()
            if self.use_raisha_attention:
                raisha_text_list = list()
            else:
                raisha_text_list = None
            raisha = row['raisha']
            for round_num in rounds:
                # use only available rounds
                if row[f'features_round_{round_num}'] is not None:
                    if self.use_raisha_attention and round_num <= raisha:
                        if self.raisha_num_features == 0:
                            self.raisha_num_features = len(row[f'features_round_{round_num}'])
                        raisha_text_list.append(ArrayField(np.array(row[f'features_round_{round_num}']),
                                                           padding_value=-1))
                    else:
                        if self.num_features == 0:
                            self.num_features = len(row[f'features_round_{round_num}'])
                            # for transformer the input dim should be // self.num_attention_heads
                            if self.use_transformer:
                                if self.use_raisha_LSTM:  # the max size of features is raisha_num_features
                                    num_features_to_use = self.raisha_num_features
                                else:
                                    num_features_to_use = self.num_features
                                check = num_features_to_use // self.num_attention_heads
                                if check * self.num_attention_heads != num_features_to_use:
                                    self.input_dim = (check + 1) * self.num_attention_heads
                                else:
                                    self.input_dim = num_features_to_use
                        if self.use_transformer:
                            extra_columns = [-1] * (self.input_dim - len(row[f'features_round_{round_num}']))
                            raisha_data = row[f'features_round_{round_num}'] + extra_columns
                        # add -1 to saifa rounds to get the same vector length
                        elif self.use_raisha_LSTM and round_num > raisha and \
                                self.raisha_num_features > self.num_features:
                            extra_columns = [-1] * (self.raisha_num_features - self.num_features)
                            raisha_data = row[f'features_round_{round_num}'] + extra_columns
                        else:
                            raisha_data = row[f'features_round_{round_num}']
                        text_list.append(ArrayField(np.array(raisha_data), padding_value=-1))
            labels = row[self._label_column]
            metadata_dict = {column: row[column] for column in metadata_columns}
            if raisha_text_list is not None and len(raisha_text_list) == 0:  # raisha = 0 and use_raisha_attention
                raisha_text_list.append(ArrayField(np.array([-1] * self.raisha_num_features), padding_value=-1))
            yield self.text_to_instance(text_list, raisha_text_list, labels, metadata_dict)


class TransformerDatasetReader(DatasetReader):
    """
    DatasetReader for LSTM models that predict for each round the DM's decision
    """
    def __init__(self,
                 features_max_size: int,
                 num_attention_heads: int=8,
                 lazy: bool = False,
                 label_column: Optional[str] = 'labels',
                 pair_ids: list = None,
                 only_raisha: bool=False) -> None:
        super().__init__(lazy)
        self._label_column = label_column
        self.num_labels = 2
        self.pair_ids = pair_ids
        self.input_dim = features_max_size
        self.only_raisha = only_raisha
        check = features_max_size // num_attention_heads
        if check * num_attention_heads != features_max_size:
            self.input_dim = (check + 1) * num_attention_heads

    @overrides
    def text_to_instance(self, saifa_text_list: List[ArrayField], raisha_text_list: List[ArrayField],
                         labels: List[str] = None, metadata: Dict=None) -> Instance:
        raisha_text_list = ListField(raisha_text_list)
        fields = {'source': raisha_text_list}

        saifa_text_list = ListField(saifa_text_list)
        fields['target'] = saifa_text_list

        if labels:
            seq_labels_field = SequenceLabelField(labels=labels, sequence_field=saifa_text_list)
            fields['seq_labels'] = seq_labels_field
            reg_labels = [0 if label == 'hotel' else 1 for label in labels]
            reg_label_field = FloatLabelField(sum(reg_labels) / len(reg_labels))
            fields['reg_labels'] = reg_label_field

        if metadata is not None:
            fields['metadata'] = MetadataField(metadata)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        """
        This function takes a filename, read the data and produces a stream of Instances
        :param str file_path: the path to the file with the data
        :return:
        """
        # Load the data
        if 'csv' in file_path:
            df = pd.read_csv(file_path)
        else:
            df = joblib.load(file_path)

        # if we run with CV we need the pair ids to use
        if self.pair_ids is not None:
            df = df.loc[df.pair_id.isin(self.pair_ids)]

        # get the reviews and label columns -> no metadata, and metadata columns
        metadata_columns = ['raisha', 'pair_id', 'sample_id']
        rounds = list(range(1, 11))  # rounds 1-10

        for i, row in tqdm.tqdm(df.iterrows()):
            raisha = row.raisha  # raisha is between 0 to 9 (the rounds in the raisha are rounds <= raisha)
            if raisha == 0:
                continue
            saifa_text_list, raisha_text_list = list(), list()
            for round_num in rounds:
                # use only available rounds
                if row[f'features_round_{round_num}'] is not None:
                    if round_num <= raisha:  # rounds in raisha
                        extra_columns = [-1] * (self.input_dim - len(row[f'features_round_{round_num}']))
                        raisha_data = row[f'features_round_{round_num}'] + extra_columns
                        raisha_text_list.append(ArrayField(np.array(raisha_data), padding_value=-1))
                    else:  # rounds in saifa
                        if self.only_raisha and round_num == raisha+1:
                            saifa_data = [100] * self.input_dim  # special vector to indicate the start of the saifa
                        else:
                            extra_columns = [-1] * (self.input_dim - len(row[f'features_round_{round_num}']))
                            saifa_data = row[f'features_round_{round_num}'] + extra_columns
                        saifa_text_list.append(ArrayField(np.array(saifa_data), padding_value=-1))
            labels = row[self._label_column]
            metadata_dict = {column: row[column] for column in metadata_columns}
            yield self.text_to_instance(saifa_text_list=saifa_text_list, raisha_text_list=raisha_text_list,
                                        labels=labels, metadata=metadata_dict)
