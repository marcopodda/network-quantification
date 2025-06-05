from .cdq import CDQ
from .enq import ENQ
from .gesn import GESN
from .lr import LR
from .params import cdq_params, enq_params, gesn_params, lr_params, wvrn_params
from .wvrn import WVRN

MODELS = {
    "enq": ENQ,
    "cdq": CDQ,
    "wvrn": WVRN,
    "gesn": GESN,
    "lr": LR,
}

PARAMS = {
    "enq": enq_params,
    "cdq": cdq_params,
    "wvrn": wvrn_params,
    "gesn": gesn_params,
    "lr": lr_params,
}


def initialize_model(model_name: str, params_index: int = 0):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found.")

    if model_name not in PARAMS:
        raise ValueError(f"Parameters for model {model_name} not found.")

    Model = MODELS[model_name]
    parameters = PARAMS[model_name]

    if params_index >= len(parameters):
        raise ValueError(f"Parameter index {params_index} out of range for model {model_name}.")

    return Model(**parameters[params_index])
