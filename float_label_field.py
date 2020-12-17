from typing import Dict, Union, Set
import logging

from overrides import overrides
import torch

from allennlp.data.fields.field import Field
from allennlp.data.vocabulary import Vocabulary
from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class FloatLabelField(Field[torch.Tensor]):
    """
    A ``FloatLabelField`` is a label, where the labels are floats.

    This field will get converted into an integer index representing the class label.

    Parameters
    ----------
    label : ``Union[str, int]``
    label_namespace : ``str``, optional (default="labels")
    """
    # Most often, you probably don't want to have OOV/PAD tokens with a LabelField, so we warn you
    # about it when you pick a namespace that will getting these tokens by default.  It is
    # possible, however, that you _do_ actually want OOV/PAD tokens with this Field.  This class
    # variable is used to make sure that we only log a single warning for this per namespace, and
    # not every time you create one of these Field objects.
    _already_warned_namespaces: Set[str] = set()

    def __init__(self,
                 label: Union[str, int],
                 label_namespace: str = 'labels',)-> None:
        self.label = label
        self._label_namespace = label_namespace
        self._label_id = label
        self._maybe_warn_for_namespace(label_namespace)

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (self._label_namespace.endswith("labels") or self._label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning("Your label namespace was '%s'. We recommend you use a namespace "
                               "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                               "default to your vocabulary.  See documentation for "
                               "`non_padded_namespaces` parameter in Vocabulary.",
                               self._label_namespace)
                self._already_warned_namespaces.add(label_namespace)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:  # pylint: disable=no-self-use
        return {}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        # pylint: disable=unused-argument,not-callable
        tensor = torch.tensor(self._label_id, dtype=torch.float)
        return tensor

    @overrides
    def empty_field(self):
        return FloatLabelField(-1, self._label_namespace)

    def __str__(self) -> str:
        return f"FloatLabelField with label: {self.label} in namespace: '{self._label_namespace}'.'"
