from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class MergeableModule(nn.Module, ABC):
    """
    An abstract base class that inherits from nn.Module.
    It specifies functions that need to be implemented in order to be compatible with the permutation and merging code.
    """

    def __init__(self):
        super(MergeableModule, self).__init__()

    def expand(self, expansion_factor: int | float | list[float] | torch.FloatTensor):
        """
        Returns a new, functionally equivalent but wider model. The appended weights and biases are all zero.
        The corresponding is_buffer flags will be set to True.
        :param expansion_factor: the factor by which to expand/widen the model (must be >1);
                                 alternatively you can provide a list or FloatTensor of length model.num_layers, which
                                 expands each layer of the model by a different factor (at least one must be >1)
        :return: the expanded model
        """
        if type(expansion_factor) is int:
            expansion_factor = float(expansion_factor)
        if type(expansion_factor) is float:
            assert expansion_factor > 1, "Expansion factor must be greater than 1.0"
            expansion_factor = [expansion_factor] * self.num_layers
        expansion_factor = torch.FloatTensor(expansion_factor)
        assert expansion_factor.min() >= 1.0, "Expansion factors <1 are not allowed"
        assert expansion_factor.max() > 1.0, "At least one expansion factor must be >1"
        return self._expand(expansion_factor)

    @abstractmethod
    def _expand(self, expansion_factor: torch.FloatTensor):
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Subclasses must have a num_layers attribute."""  # TODO: check if really necessary
        pass

    @property
    def num_params(self) -> int:
        """The number of parameters in the module (buffer_flag is ignored)."""
        return sum([v.numel() for k, v in self.state_dict().items() if "is_buffer" not in k])
