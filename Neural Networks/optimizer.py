"""
This class contains helper functions which will help get the optimizer
"""

from typing import Any, Dict

import torch


def get_optimizer(
    model: torch.nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Returns the optimizer initializer according to the config

    Note: config has a minimum of three entries.
    Feel free to add more entries if you want.
    But do not change the name of the three existing entries

    Args:
    - model: the model to optimize for
    - config: a dictionary containing parameters for the config
    Returns:
    - optimizer: the optimizer
    """

    optimizer = None

    optimizer_type = config.get("optimizer_type", "sgd")
    learning_rate = config.get("lr", 1e-2)
    weight_decay = config.get("weight_decay", 1e-5)

    ############################################################################
    # Student code begin
    ############################################################################
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        momentum = config.get("momentum", 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer

    ############################################################################
    # Student code end
    ############################################################################

