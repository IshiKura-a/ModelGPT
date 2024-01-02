from torch import nn
from torchvision.models import get_model, get_model_weights


def get_pretrained_model(name: str, num_classes: int) -> nn.Module:
    model = get_model(name, weights=list(get_model_weights(name))[-1])
    if 'mobilenet' in name or 'efficientnet' in name:
        d = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(d, num_classes)
    elif 'resnet' in name or 'shufflenet':
        d = model.fc.in_features
        model.fc = nn.Linear(d, num_classes)
    else:
        raise NotImplementedError

    return model
