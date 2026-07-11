import torch
from aegl.data import create_dataloaders


def test_loader_contract():
    loaders = create_dataloaders(batch_size=2, num_workers=0, max_samples=8)
    images, labels = next(iter(loaders["train"]))
    assert images.shape[0] == 2
    assert labels.shape == (2,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
