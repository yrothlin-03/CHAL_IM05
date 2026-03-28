# from .dataset import get_loaders as test
from .dataloader import get_loaders
from .contrastive_dataset import get_contrastive_loaders

__all__ = ["get_loaders", "get_contrastive_loaders"]