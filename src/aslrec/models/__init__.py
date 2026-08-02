from .asl_cnn import ASL_CNN
from .slrvgg8 import SLRVGG8

MODEL_REGISTRY = {
    "slrvgg8": SLRVGG8,
    "asl_cnn": ASL_CNN,
}


def build_model(name: str, num_classes: int):
    try:
        return MODEL_REGISTRY[name](num_classes=num_classes)
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Options: {sorted(MODEL_REGISTRY)}")
