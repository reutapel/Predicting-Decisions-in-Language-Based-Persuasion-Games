from transformers.modeling_bert import BertLMPredictionHead, BertPreTrainedModel, BertModel
from BERT.bert_text_dataset import BertTextDataset
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
import logging


class BertForOneFeaturePredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooler = BertForOneFeaturePredictionHead.masked_avg_pooler
        self.decoder = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, sequence_mask):
        pooled_output = self.pooler(sequence_output, sequence_mask)

        output = self.decoder(pooled_output)
        return output

    @staticmethod
    def masked_avg_pooler(sequences: torch.Tensor, masks: torch.Tensor = None) -> torch.Tensor:
        if masks is None:
            return sequences.mean(dim=1)
        masked_sequences = sequences * masks.float().unsqueeze(dim=-1).expand_as(sequences)
        sequence_lengths = masks.sum(dim=-1).view(-1, 1, 1).expand_as(sequences)
        return torch.sum(masked_sequences / sequence_lengths, dim=1)


class BertForFourFeaturesPreTrainingHeads(nn.Module):
    def __init__(self, config, num_features):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.num_features = num_features
        self.feature_predictions = nn.ModuleDict()

        for feature in range(num_features):
            self.feature_predictions[f'feature_{feature+1}_prediction'] = BertForOneFeaturePredictionHead(config)

    def forward(self, sequence_output, sequence_mask):
        lm_prediction_scores = self.predictions(sequence_output)
        feature_prediction_scores = dict()
        for key, layer in self.feature_predictions.items():
            feature_prediction_scores[f'{key}_score'] = layer(sequence_output, sequence_mask)

        return lm_prediction_scores, feature_prediction_scores


class BertForFourFeaturesPreTraining(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **feature_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing each feature prediction (classification) loss.
            Indices should be in ``[0, 1]``.
            ``0`` indicates that the feature doesn't presented in the text,
            ``1`` indicates that the feature presented in the text,

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **lm_prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    """
    def __init__(self, config, num_features):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertForFourFeaturesPreTrainingHeads(config, num_features)

        self.init_weights()
        self.tie_weights()
        self.num_features = num_features

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, masked_lm_labels=None, features_labels_dict: dict=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        sequence_output, pooled_output = outputs[:2]
        lm_prediction_scores, feature_prediction_scores = self.cls(sequence_output, attention_mask)
        outputs = (lm_prediction_scores, feature_prediction_scores,) + outputs[2:]

        loss_fct = CrossEntropyLoss(ignore_index=BertTextDataset.MLM_IGNORE_LABEL_IDX)
        loss_fct_per_sample = CrossEntropyLoss(ignore_index=BertTextDataset.MLM_IGNORE_LABEL_IDX, reduction='none')
        if masked_lm_labels is not None and features_labels_dict is not None:
            masked_lm_loss = loss_fct(lm_prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            loss = masked_lm_loss
            feature_losses = dict()
            features_loss_fct_per_sample = dict()
            for i, key in enumerate(feature_prediction_scores.keys()):
                feature_losses[f'feature_{i+1}_loss'] =\
                    loss_fct(feature_prediction_scores[f'feature_{i+1}_prediction_score'].view(-1, 2),
                             features_labels_dict[f'feature_{i+1}_label'].view(-1))
                loss += feature_losses[f'feature_{i+1}_loss']
                features_loss_fct_per_sample[f'feature_{i+1}_loss'] =\
                    loss_fct_per_sample(feature_prediction_scores[f'feature_{i+1}_prediction_score'],
                                        features_labels_dict[f'feature_{i+1}_label'])
            outputs = (loss,
                       torch.stack([loss_fct_per_sample(lm_prediction_scores.view(-1, self.config.vocab_size),
                                                        masked_lm_labels.view(-1))
                                   .view_as(masked_lm_labels)[i, :].masked_select(
                           masked_lm_labels[i, :] > -1).mean() for i in range(masked_lm_labels.size(0))]),
                       features_loss_fct_per_sample,) + outputs

        return outputs
