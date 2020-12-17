import torch
import torch.nn as nn
from typing import *
from allennlp.models import Model
from allennlp.data.vocabulary import Vocabulary
import torch.nn.functional as F
import pandas as pd
import numpy as np
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from collections import defaultdict
from allennlp.modules.attention.dot_product_attention import DotProductAttention
from allennlp.modules.attention.attention import Attention
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder,\
    TransformerDecoderLayer
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_
from allennlp.nn.util import add_positional_features
from allennlp.data.iterators import DataIterator
from tqdm import tqdm
from allennlp.data import Instance
from allennlp.nn import util as nn_util
from allennlp.nn.util import masked_softmax
import copy


def save_predictions_seq_models(prediction_df: dict, predictions: torch.Tensor, gold_labels: torch.Tensor,
                                metadata, epoch: int, is_train: bool, mask: torch.Tensor, use_raisha_lstm: bool=False) \
        -> dict:
    """
    This function get the predictions class and save with the gold labels for sequence models where each sample had a
    list of predictions and labels
    :param prediction_df: dataframe with all the predictions by now
    :param predictions: the predictions for a specific batch in a specific epoch
    :param gold_labels: the labels for a specific batch
    :param metadata: the metadata for a specific batch
    :param epoch: the number of epoch
    :param is_train: if this is train data or not
    :param mask: mask for each sample- which index is a padding
    :param use_raisha_lstm: these models labels are upper Triangular matrix
    :return:
    """

    for i in range(predictions.shape[0]):  # go over each sample
        if epoch == 0:  # if this is the first epoch - keep the label
            if use_raisha_lstm:
                gold_labels_list = gold_labels[i][(mask[i] != 0).nonzero().T[0]].tolist()
            else:
                gold_labels_list = gold_labels[i][:mask[i].sum()].tolist()
            # if hotel_label_0:
            #     gold_labels_list = [0 if label == 1 else 1 for label in gold_labels_list]
            prediction_df[metadata[i]['sample_id']] = \
                {'is_train': is_train, f'labels': gold_labels_list,
                 f'total_payoff_label': sum(gold_labels_list) / len(gold_labels_list)}
            if 'raisha' in metadata[i].keys():
                prediction_df[metadata[i]['sample_id']]['raisha'] = metadata[i]['raisha']
        # most_frequent_mask will have 1 for every trial that the prediction is 0.5, 0.5 --> the model did not decide
        temp_prediction_list = predictions[i][(mask[i] != 0).nonzero().T[0]]
        most_frequent_tensor = torch.tensor([0.5, 0.5])
        if torch.cuda.is_available():
            most_frequent_tensor = most_frequent_tensor.cuda()
        most_frequent_mask = torch.tensor(~((most_frequent_tensor == temp_prediction_list).sum(1) == 2),
                                          dtype=int).tolist()
        predictions_list = temp_prediction_list.argmax(1).tolist()
        predictions_list = (np.array(predictions_list) * np.array(most_frequent_mask)).tolist()
        # if hotel_label_0:
        #     predictions_list = [0 if prediction == 1 else 1 for prediction in predictions_list]
        prediction_df[metadata[i]['sample_id']][f'predictions_{epoch}'] = predictions_list
        prediction_df[metadata[i]['sample_id']][f'total_payoff_prediction_{epoch}'] =\
            sum(predictions_list) / len(predictions_list)

    # prediction_df_temp = pd.DataFrame.from_dict(predictions_dict, orient='index')
    #
    # if epoch != 0:
    #     prediction_df = prediction_df.merge(prediction_df_temp, how='left', right_index=True, left_index=True)
    # else:
    #     prediction_df = pd.concat([prediction_df, prediction_df_temp])

    return prediction_df


def save_predictions(prediction_df: pd.DataFrame, predictions: torch.Tensor, gold_labels: torch.Tensor, metadata,
                     epoch: int, is_train: bool, int_label: bool=True) -> pd.DataFrame:
    """
    This function get the predictions class and save with the gold labels
    :param prediction_df: dataframe with all the predictions by now
    :param predictions: the predictions for a specific batch in a specific epoch
    :param gold_labels: the labels for a specific batch
    :param metadata: the metadata for a specific batch
    :param epoch: the number of epoch
    :param is_train: if this is train data or not
    :param int_label: if the label type is int
    :return:
    """

    metadata_df = pd.DataFrame(metadata)
    # Create a data frame with the sample ID, the prediction, the label and if the prediction is correct
    if int_label:
        gold_labels = gold_labels.view(gold_labels.shape[0], -1).argmax(1)
        predictions = predictions.view(predictions.shape[0], -1).argmax(1)
    else:
        predictions = predictions.view(predictions.shape[0]).tolist()
    label_prediction = \
        pd.concat([metadata_df.sample_id,
                   pd.DataFrame(gold_labels, columns=['total_payoff_label']),
                   pd.DataFrame(predictions, columns=[f'prediction_{epoch}'])], axis=1)
    if 'raisha' in metadata_df.columns:
        label_prediction = label_prediction.merge(metadata_df[['sample_id', 'raisha']], how='left', on='sample_id')
    label_prediction[f'correct_{epoch}'] =\
        np.where(label_prediction[f'prediction_{epoch}'] == label_prediction.total_payoff_label, 1, 0)
    if epoch == 0:  # if this is the first epoch - keep the label
        train = pd.DataFrame(pd.Series([is_train], name='is_train').repeat(label_prediction.shape[0]))
        train.index = label_prediction.index
        label_prediction = pd.concat([train, label_prediction], axis=1)
        prediction_df = pd.concat([prediction_df, label_prediction])

    else:  # if this is not the first label drop the label
        label_prediction = label_prediction.drop(['total_payoff_label', 'raisha'], axis=1)
        prediction_df = prediction_df.merge(label_prediction, how='left', on='sample_id')
        if f'prediction_{epoch}_x' in prediction_df.columns:
            prediction_df[f'prediction_{epoch}_x'] = np.where(prediction_df[f'prediction_{epoch}_x'].isnull(),
                                                              prediction_df[f'prediction_{epoch}_y'],
                                                              prediction_df[f'prediction_{epoch}_x'])
            prediction_df[f'correct_{epoch}_x'] = np.where(prediction_df[f'correct_{epoch}_x'].isnull(),
                                                           prediction_df[f'correct_{epoch}_y'],
                                                           prediction_df[f'correct_{epoch}_x'])
            prediction_df = prediction_df.drop([f'correct_{epoch}_y', f'prediction_{epoch}_y'], axis=1)
            prediction_df.rename(columns={f'correct_{epoch}_x': f'correct_{epoch}',
                                          f'prediction_{epoch}_x': f'prediction_{epoch}'}, inplace=True)

    return prediction_df


class LSTMBasedModel(Model):
    """
    This is a LSTM model that predict the class for each 't' and the average total payoff of the saifa (2 losses)
    """
    def __init__(self,
                 encoder: Seq2SeqEncoder,
                 metrics_dict_seq: dict,
                 metrics_dict_reg: dict,
                 vocab: Vocabulary,
                 attention: Attention = DotProductAttention(),
                 seq_weight_loss: float=1.0,
                 reg_weight_loss: float=1.0,
                 reg_seq_weight_loss: float=1.0,
                 predict_seq: bool=True,
                 predict_avg_total_payoff: bool=True,
                 batch_size: int=10,
                 linear_dim=None,
                 dropout: float=None,
                 use_last_hidden_vec: bool=False,
                 use_transformer_encode: bool=False,
                 input_dim: int=0,
                 use_raisha_attention: bool=False,
                 raisha_num_features: int=0,
                 linear_layers_activation='relu',
                 use_raisha_LSTM: bool=False) -> None:
        super(LSTMBasedModel, self).__init__(vocab)
        self.encoder = encoder
        self.use_raisha_LSTM = use_raisha_LSTM
        if use_transformer_encode:
            encoder_output_dim = input_dim
        else:
            encoder_output_dim = encoder.get_output_dim()

        self.use_raisha_LSTM = use_raisha_LSTM

        if use_raisha_attention and raisha_num_features > 0:  # add attention layer to create raisha representation
            self.raisha_attention_layer = attention
            self.raisha_attention_vector = torch.randn((batch_size, raisha_num_features), requires_grad=True)
            # linear layer: raisha num features -> saifa num features (encoder.get_input_dim())
            self.linear_after_raisha_attention_layer = LinearLayer(input_size=raisha_num_features,
                                                                   output_size=encoder.get_input_dim(),
                                                                   activation=linear_layers_activation)
            if torch.cuda.is_available():
                self.raisha_attention_vector = self.raisha_attention_vector.cuda()
        else:
            self.raisha_attention_layer = None
            self.raisha_attention_vector = None

        if predict_seq:  # need hidden2tag layer
            if linear_dim is not None:  # add linear layer before hidden2tag
                self.linear_layer = LinearLayer(input_size=encoder_output_dim, output_size=linear_dim, dropout=dropout,
                                                activation=linear_layers_activation)
                hidden2tag_input_size = linear_dim
            else:
                self.linear_layer = None
                hidden2tag_input_size = encoder_output_dim
            self.hidden2tag = LinearLayer(input_size=hidden2tag_input_size, output_size=vocab.get_vocab_size('labels'),
                                          dropout=dropout, activation=linear_layers_activation)

        if predict_avg_total_payoff:  # need attention and regression layer
            self.attention = attention
            self.linear_after_attention_layer = LinearLayer(input_size=encoder_output_dim, output_size=batch_size,
                                                            activation=linear_layers_activation)
            self.regressor = LinearLayer(input_size=batch_size, output_size=1, activation=linear_layers_activation)
            self.attention_vector = torch.randn((batch_size, encoder_output_dim), requires_grad=True)
            if torch.cuda.is_available():
                self.attention_vector = self.attention_vector.cuda()
            self.mse_loss = nn.MSELoss()

        if predict_avg_total_payoff and predict_seq:  # for avg_turn models
            self.seq_reg_mse_loss = nn.MSELoss()

        if use_last_hidden_vec:
            if linear_dim is not None:  # add linear layer before last_hidden_reg
                self.linear_layer = LinearLayer(input_size=encoder_output_dim, output_size=linear_dim, dropout=dropout)
                self.last_hidden_reg = LinearLayer(input_size=linear_dim, output_size=1, dropout=dropout)
            else:
                self.linear_layer = None
                self.last_hidden_reg = LinearLayer(input_size=encoder_output_dim, output_size=1, dropout=dropout)

        self.metrics_dict_seq = metrics_dict_seq
        self.metrics_dict_reg = metrics_dict_reg
        self.seq_predictions = defaultdict(dict)
        self.reg_predictions = pd.DataFrame()
        self._epoch = 0
        self._first_pair = None
        self.seq_weight_loss = seq_weight_loss
        self.reg_weight_loss = reg_weight_loss
        self.reg_seq_weight_loss = reg_seq_weight_loss
        self.predict_seq = predict_seq
        self.predict_avg_total_payoff = predict_avg_total_payoff
        self.use_last_hidden_vec = use_last_hidden_vec

    def forward(self,
                sequence_review: torch.Tensor,
                metadata: dict,
                raisha_sequence_review: torch.Tensor = None,
                seq_labels: torch.Tensor=None,
                reg_labels: torch.Tensor=None) -> Dict[str, torch.Tensor]:

        if self._first_pair is not None:
            if self._first_pair == metadata[0]['pair_id']:
                self._epoch += 1
        else:
            self._first_pair = metadata[0]['pair_id']

        output = dict()
        mask = get_text_field_mask({'tokens': sequence_review})
        if torch.cuda.is_available():  # change to cuda
            mask = mask.cuda()
            sequence_review = sequence_review.cuda()
            if seq_labels is not None:
                seq_labels = seq_labels.cuda()
            if reg_labels is not None:
                reg_labels = reg_labels.cuda()

        if self.raisha_attention_layer is not None:
            raisha_mask = get_text_field_mask({'tokens': raisha_sequence_review})
            # (batch_size, max_raisha_len, raisha_dimensions) * (batch_size, raisha_dimensions, 1) ->
            # (batch_size, max_raisha_len)
            raisha_attention_outpus = self.raisha_attention_layer(self.raisha_attention_vector, raisha_sequence_review,
                                                                  raisha_mask)
            # (batch_size, 1, max_raisha_len) * (batch_size, max_raisha_len, raisha_dimensions) ->
            # (batch_size, raisha_dimensions)
            attention_output = torch.bmm(raisha_attention_outpus.unsqueeze(1), raisha_sequence_review).squeeze()
            # (batch_size, raisha_dimensions) -> (batch_size, saifa_dimensions)
            raisha_representation = self.linear_after_raisha_attention_layer(attention_output)
            all_raisha_saifa_representations = list()
            for seq_id in range(sequence_review.shape[0]):
                seq_raisha_saifa_representations = \
                    torch.cat([raisha_representation[seq_id].unsqueeze(0), sequence_review[seq_id]], dim=0)
                all_raisha_saifa_representations.append(seq_raisha_saifa_representations)
            sequence_review = torch.stack(all_raisha_saifa_representations, dim=0)
            # create the mask with all the first enter with 1 --> for the raisha representation
            ones_tensor = torch.ones(mask.shape[0]).type(torch.LongTensor).unsqueeze(0)
            original_mask = copy.deepcopy(mask)
            if torch.cuda.is_available():  # change to cuda
                ones_tensor = ones_tensor.cuda()
            mask = torch.cat((ones_tensor.T, mask), dim=1)
            mask[0][0] = 0  # for raisha 0 --> the first column is not really the raisha representation

        encoder_out = self.encoder(sequence_review, mask)
        # remove raisha hidden vectors from encoder_out and mask
        if self.raisha_attention_layer is not None:
            encoder_out = encoder_out[:, 1:, :]
            mask = original_mask

        if self.use_raisha_LSTM:  # change mask to cover the raisha rounds
            for seq_id in range(encoder_out.shape[0]):
                seq_raisha = metadata[seq_id]['raisha']
                mask[seq_id, :seq_raisha] = 0
            # for use raisha_LSTM --> the labels should look the same as the mask and results
            seq_labels = seq_labels[0].repeat(10, 1) * mask

        if self.predict_seq:
            if self.linear_layer is not None:
                encoder_out_linear = self.linear_layer(encoder_out)  # add linear layer before hidden2tag
                decision_logits = self.hidden2tag(encoder_out_linear)
            else:
                decision_logits = self.hidden2tag(encoder_out)
            output['decision_logits'] = masked_softmax(decision_logits, mask.unsqueeze(2))
            self.seq_predictions = save_predictions_seq_models(prediction_df=self.seq_predictions, mask=mask,
                                                               predictions=output['decision_logits'],
                                                               gold_labels=seq_labels, metadata=metadata,
                                                               epoch=self._epoch, is_train=self.training,
                                                               use_raisha_lstm=self.use_raisha_LSTM)

        if self.predict_avg_total_payoff:
            if self.use_last_hidden_vec:
                max_raisha = metadata[-1]['raisha']
                all_last_hidden_vectors = list()
                for seq_id in range(encoder_out.shape[0]):
                    seq_raisha = metadata[seq_id]['raisha']
                    last_hidden_vec = encoder_out[seq_id][max_raisha - seq_raisha]
                    all_last_hidden_vectors.append(last_hidden_vec)
                all_last_hidden_vectors = torch.stack(all_last_hidden_vectors, dim=0)
                if self.linear_layer is not None:
                    all_last_hidden_vectors = self.linear_layer(all_last_hidden_vectors)
                regression_output = self.last_hidden_reg(all_last_hidden_vectors)
            else:
                # (batch_size, seq_len, dimensions) * (batch_size, dimensions, 1) -> (batch_size, seq_len)
                attention_output = self.attention(self.attention_vector, encoder_out, mask)
                # (batch_size, 1, seq_len) * (batch_size, seq_len, dimensions) -> (batch_size, dimensions)
                attention_output = torch.bmm(attention_output.unsqueeze(1), encoder_out).squeeze()
                # (batch_size, dimensions) -> (batch_size, batch_size)
                linear_out = self.linear_after_attention_layer(attention_output)
                # (batch_size, batch_size) -> (batch_size, 1)
                regression_output = self.regressor(linear_out)
            output['regression_output'] = regression_output
            self.reg_predictions = save_predictions(prediction_df=self.reg_predictions,
                                                    predictions=output['regression_output'], gold_labels=reg_labels,
                                                    metadata=metadata, epoch=self._epoch, is_train=self.training,
                                                    int_label=False)

        if seq_labels is not None or reg_labels is not None:
            temp_loss = 0
            if self.predict_seq and seq_labels is not None:
                for metric_name, metric in self.metrics_dict_seq.items():
                    metric(decision_logits, seq_labels, mask)
                output['seq_loss'] = sequence_cross_entropy_with_logits(decision_logits, seq_labels, mask)
                temp_loss += self.seq_weight_loss * output['seq_loss']
            if self.predict_avg_total_payoff and reg_labels is not None:
                for metric_name, metric in self.metrics_dict_reg.items():
                    metric(regression_output, reg_labels.unsqueeze(1))
                output['reg_loss'] = self.mse_loss(regression_output, reg_labels.view(reg_labels.shape[0], -1))
                temp_loss += self.reg_weight_loss * output['reg_loss']
            # loss for avg_turn models
            if self.predict_seq and seq_labels is not None and self.predict_avg_total_payoff and reg_labels is not None:
                proportion_from_decisions = decision_logits.detach().clone()
                proportion_from_decisions = masked_softmax(proportion_from_decisions, mask.unsqueeze(2))
                proportion_from_decisions = proportion_from_decisions.max(2)[1] * mask
                proportion_from_decisions = proportion_from_decisions.sum(1).float() / mask.sum(1)
                output['reg_seq_loss'] = \
                    self.seq_reg_mse_loss(regression_output, proportion_from_decisions.view(reg_labels.shape[0], -1))
                temp_loss += self.reg_seq_weight_loss * output['reg_seq_loss']
            output['loss'] = temp_loss

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        merge the 2 metrics to get all the metrics
        :param reset:
        :return:
        """
        return_metrics = dict()
        if self.predict_seq:
            seq_metrics = dict()
            for metric_name, metric in self.metrics_dict_seq.items():
                if metric_name == 'F1measure_hotel_label':
                    seq_metrics['precision_hotel_label'], seq_metrics['recall_hotel_label'],\
                        seq_metrics['fscore_hotel_label'] = metric.get_metric(reset)
                elif metric_name == 'F1measure_home_label':
                    seq_metrics['precision_home_label'], seq_metrics['recall_home_label'],\
                        seq_metrics['fscore_home_label'] = metric.get_metric(reset)
                else:
                    seq_metrics[metric_name] = metric.get_metric(reset)
            return_metrics.update(seq_metrics)
        if self.predict_avg_total_payoff:
            reg_metrics =\
                {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics_dict_reg.items()}
            return_metrics.update(reg_metrics)
        return return_metrics


class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size, dropout=None, activation: str='relu'):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        if type(dropout) is float and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            self.activation = False

    def forward(self, x):
        linear_out = x
        if self.dropout is not None:
            linear_out = self.dropout(linear_out)
        linear_out = self.linear(linear_out)
        if self.activation:
            linear_out = self.activation(linear_out)
        return linear_out


class TransformerBasedModel(Model):
    """Implement encoder-decoder transformer. The architecture
    is based on the paper "Attention Is All You Need". Ashish Vaswani, Noam Shazeer,
    Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and
    Illia Polosukhin. 2017. Attention is all you need. """
    def __init__(self,
                 vocab: Vocabulary,
                 metrics_dict_seq: dict,
                 metrics_dict_reg: dict,
                 input_dim=512,
                 num_attention_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 feedforward_hidden_dim=2048,
                 dropout=0.1,
                 transformer_dropout=0.1,
                 activation='relu',
                 linear_layers_activation='relu',
                 custom_encoder=None,
                 custom_decoder=None,
                 positional_encoding: Optional[str] = None,
                 predict_avg_total_payoff: bool=True,
                 predict_seq: bool = True,
                 attention: Attention = DotProductAttention(),
                 seq_weight_loss: float = 0.5,
                 reg_weight_loss: float = 0.5,
                 batch_size: int = 9,
                 linear_dim: int=None,
                 only_raisha: bool=False,  # if not saifa input is given
                 ):
        super(TransformerBasedModel, self).__init__(vocab)

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(input_dim, num_attention_heads, feedforward_hidden_dim,
                                                    transformer_dropout, activation)
            encoder_norm = LayerNorm(input_dim)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(input_dim, num_attention_heads, feedforward_hidden_dim,
                                                    transformer_dropout, activation)
            decoder_norm = LayerNorm(input_dim)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self._reset_parameters()

        self._input_dim = input_dim
        self.num_attention_heads = num_attention_heads

        if positional_encoding is None:
            self._sinusoidal_positional_encoding = False
            self._positional_embedding = None
        elif positional_encoding == "sinusoidal":
            self._sinusoidal_positional_encoding = True
            self._positional_embedding = None
        else:
            raise ValueError(
                "positional_encoding must be one of None, 'sinusoidal', or 'embedding'"
            )

        if predict_avg_total_payoff:  # need attention and regression layer
            self.attention = attention
            if linear_dim is not None and predict_seq:  # avg_turn_linear models
                input_dim_attention = linear_dim
            else:
                input_dim_attention = input_dim
            self.linear_after_attention_layer = LinearLayer(input_size=input_dim_attention, output_size=batch_size,
                                                            activation=linear_layers_activation)
            self.regressor = LinearLayer(input_size=batch_size, output_size=1, dropout=dropout,
                                         activation=linear_layers_activation)
            self.attention_vector = torch.randn((batch_size, input_dim_attention), requires_grad=True)
            if torch.cuda.is_available():
                self.attention_vector = self.attention_vector.cuda()
            self.mse_loss = nn.MSELoss()

        if predict_seq:  # need hidden2tag layer
            if linear_dim is not None:  # add linear layer before hidden2tag
                self.linear_layer = LinearLayer(input_size=input_dim, output_size=linear_dim, dropout=dropout,
                                                activation=linear_layers_activation)
                hidden2tag_input_size = linear_dim
            else:
                self.linear_layer = None
                hidden2tag_input_size = input_dim
            self.hidden2tag = LinearLayer(input_size=hidden2tag_input_size, output_size=vocab.get_vocab_size('labels'),
                                          dropout=dropout, activation=linear_layers_activation)

        self.metrics_dict_seq = metrics_dict_seq
        self.metrics_dict_reg = metrics_dict_reg
        self.seq_predictions = defaultdict(dict)
        self.reg_predictions = pd.DataFrame()
        self._epoch = 0
        self._first_pair = None
        self.seq_weight_loss = seq_weight_loss
        self.reg_weight_loss = reg_weight_loss
        self.predict_avg_total_payoff = predict_avg_total_payoff
        self.predict_seq = predict_seq
        self.only_raisha = only_raisha

    def get_input_dim(self) -> int:
        return self._input_dim

    def get_output_dim(self) -> int:
        return 1

    def forward(self, source: torch.Tensor, target: torch.Tensor, metadata: dict, seq_labels: torch.Tensor = None,
                reg_labels: torch.Tensor = None, source_mask: Optional[torch.Tensor]=None,
                target_mask: Optional[torch.Tensor]=None, memory_mask: Optional[torch.Tensor]=None,
                src_key_padding_mask: Optional[torch.Tensor]=None, tgt_key_padding_mask: Optional[torch.Tensor]=None,
                memory_key_padding_mask: Optional[torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        r"""Take in and process masked source/target sequences.
        Args:
            source: the sequence to the encoder (required).
            target: the sequence to the decoder (required).
            metadata: the metadata of the samples (required).
            seq_labels: the labels of each round (optional).
            reg_labels: the labels of the total future payoff (optional).
            source_mask: the additive mask for the src sequence (optional).
            target_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the ByteTensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the ByteTensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the ByteTensor mask for memory keys per batch (optional).
        Shape:
            - source: :math:`(S, N, E)`.
            - target: :math:`(T, N, E)`.
            - source_mask: :math:`(S, S)`.
            - target_mask: :math:`(T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(N, S)`.
            Note: [src/tgt/memory]_mask should be filled with
            float('-inf') for the masked positions and float(0.0) else. These masks
            ensure that predictions for position i depend only on the unmasked positions
            j and are applied identically for each sequence in a batch.
            [src/tgt/memory]_key_padding_mask should be a ByteTensor where True values are positions
            that should be masked with float('-inf') and False values will be unchanged.
            This mask ensures that no information will be taken from position i if
            it is masked, and has a separate mask for each sequence in a batch.
            - output: :math:`(T, N, E)`.
            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decode.
            where S is the source sequence length, T is the target sequence length, N is the
            batch size, E is the feature number
        """

        if self._first_pair is not None:
            if self._first_pair == metadata[0]['pair_id']:
                self._epoch += 1
        else:
            self._first_pair = metadata[0]['pair_id']

        output = dict()
        src_key_padding_mask = get_text_field_mask({'tokens': source})
        tgt_key_padding_mask = get_text_field_mask({'tokens': target})
        # The torch transformer takes the mask backwards.
        src_key_padding_mask_byte = ~src_key_padding_mask.bool()
        tgt_key_padding_mask_byte = ~tgt_key_padding_mask.bool()
        # create mask where only the first round is not masked --> need to be the same first round for all sequances
        if self.only_raisha:
            temp_mask = torch.ones(tgt_key_padding_mask_byte.shape, dtype=torch.bool)
            temp_mask[:, 0] = False
            tgt_key_padding_mask_byte = temp_mask

        if self._sinusoidal_positional_encoding:
            source = add_positional_features(source)
            target = add_positional_features(target)

        if torch.cuda.is_available():  # change to cuda
            source = source.cuda()
            target = target.cuda()
            tgt_key_padding_mask = tgt_key_padding_mask.cuda()
            src_key_padding_mask = src_key_padding_mask.cuda()
            tgt_key_padding_mask_byte = tgt_key_padding_mask_byte.cuda()
            src_key_padding_mask_byte = src_key_padding_mask_byte.cuda()
            if seq_labels is not None:
                seq_labels = seq_labels.cuda()
            if reg_labels is not None:
                reg_labels = reg_labels.cuda()

        # The torch transformer expects the shape (sequence, batch, features), not the more
        # familiar (batch, sequence, features), so we have to fix it.
        source = source.permute(1, 0, 2)
        target = target.permute(1, 0, 2)

        if source.size(1) != target.size(1):
            raise RuntimeError("the batch number of src and tgt must be equal")

        if source.size(2) != self._input_dim or target.size(2) != self._input_dim:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        encoder_out = self.encoder(source, src_key_padding_mask=src_key_padding_mask_byte)
        decoder_output = self.decoder(target, encoder_out, tgt_key_padding_mask=tgt_key_padding_mask_byte,
                                      memory_key_padding_mask=src_key_padding_mask_byte)
        decoder_output = decoder_output.permute(1, 0, 2)

        if self.predict_seq:
            if self.linear_layer is not None:
                decoder_output = self.linear_layer(decoder_output)  # add linear layer before hidden2tag
            decision_logits = self.hidden2tag(decoder_output)
            output['decision_logits'] = masked_softmax(decision_logits, tgt_key_padding_mask.unsqueeze(2))
            self.seq_predictions = save_predictions_seq_models(prediction_df=self.seq_predictions,
                                                               mask=tgt_key_padding_mask,
                                                               predictions=output['decision_logits'],
                                                               gold_labels=seq_labels, metadata=metadata,
                                                               epoch=self._epoch, is_train=self.training,)

        if self.predict_avg_total_payoff:
            # (batch_size, seq_len, dimensions) * (batch_size, dimensions, 1) -> (batch_size, seq_len)
            attention_output = self.attention(self.attention_vector, decoder_output, tgt_key_padding_mask)
            # (batch_size, 1, seq_len) * (batch_size, seq_len, dimensions) -> (batch_size, dimensions)
            attention_output = torch.bmm(attention_output.unsqueeze(1), decoder_output).squeeze()
            # (batch_size, dimensions) -> (batch_size, batch_size)
            linear_out = self.linear_after_attention_layer(attention_output)
            # (batch_size, batch_size) -> (batch_size, 1)
            regression_output = self.regressor(linear_out)
            output['regression_output'] = regression_output
            self.reg_predictions = save_predictions(prediction_df=self.reg_predictions,
                                                    predictions=output['regression_output'], gold_labels=reg_labels,
                                                    metadata=metadata, epoch=self._epoch, is_train=self.training,
                                                    int_label=False)

        if seq_labels is not None or reg_labels is not None:
            temp_loss = 0
            if self.predict_seq and seq_labels is not None:
                for metric_name, metric in self.metrics_dict_seq.items():
                    metric(decision_logits, seq_labels, tgt_key_padding_mask)
                output['seq_loss'] = sequence_cross_entropy_with_logits(decision_logits, seq_labels,
                                                                        tgt_key_padding_mask)
                temp_loss += self.seq_weight_loss * output['seq_loss']
            if self.predict_avg_total_payoff and reg_labels is not None:
                for metric_name, metric in self.metrics_dict_reg.items():
                    metric(regression_output, reg_labels, tgt_key_padding_mask)
                output['reg_loss'] = self.mse_loss(regression_output, reg_labels.view(reg_labels.shape[0], -1))
                temp_loss += self.reg_weight_loss * output['reg_loss']

            output['loss'] = temp_loss

        return output

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """
        merge the 2 metrics to get all the metrics
        :param reset:
        :return:
        """
        return_metrics = dict()
        if self.predict_seq:
            seq_metrics = dict()
            for metric_name, metric in self.metrics_dict_seq.items():
                if metric_name == 'F1measure_hotel_label':
                    seq_metrics['precision_hotel_label'], seq_metrics['recall_hotel_label'],\
                        seq_metrics['fscore_hotel_label'] = metric.get_metric(reset)
                elif metric_name == 'F1measure_home_label':
                    seq_metrics['precision_home_label'], seq_metrics['recall_home_label'],\
                        seq_metrics['fscore_home_label'] = metric.get_metric(reset)
                else:
                    seq_metrics[metric_name] = metric.get_metric(reset)
            return_metrics.update(seq_metrics)
        if self.predict_avg_total_payoff:
            reg_metrics =\
                {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics_dict_reg.items()}
            return_metrics.update(reg_metrics)
        return return_metrics


def tonp(tsr): return tsr.detach().cpu().numpy()


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator,
                 cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device
        self.seq_predictions = defaultdict(dict)
        self.reg_predictions = pd.DataFrame()

    def _extract_data(self, batch):
        out_dict = self.model(**batch)
        if 'regression_output' in out_dict:
            self.reg_predictions = save_predictions(prediction_df=self.reg_predictions,
                                                    predictions=out_dict['regression_output'],
                                                    gold_labels=batch['reg_labels'],
                                                    metadata=batch['metadata'], epoch=0, is_train=False,
                                                    int_label=False)
            return
        elif 'decision_logits' in out_dict:
            if 'sequence_review' in batch:
                mask = get_text_field_mask({'tokens': batch['sequence_review']})
            elif 'target' in batch:
                mask = get_text_field_mask({'tokens': batch['target']})
            else:
                print('Error: either sequence_review or target need to be in batch')
                raise TypeError
            self.seq_predictions = save_predictions_seq_models(prediction_df=self.seq_predictions, mask=mask,
                                                               predictions=out_dict['decision_logits'],
                                                               gold_labels=batch['seq_labels'],
                                                               metadata=batch['metadata'],
                                                               epoch=0, is_train=False)
            return
        else:
            print('Error: regression_output and decision_logits are not in out_dict keys')
            raise TypeError

    def predict(self, ds: Iterable[Instance]):
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator, total=self.iterator.get_num_batches(ds))
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                self._extract_data(batch)

        return
