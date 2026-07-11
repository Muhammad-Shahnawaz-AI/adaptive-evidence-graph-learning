import torch
from aegl.config import AEGLConfig
from aegl.data import build_dataloaders


def test_loader_contract():
    cfg = AEGLConfig()
    train_loader, id_loader, ood_loader = build_dataloaders(
        cfg, samples_per_class=2, batch_size=2, num_workers=0
    )
    images, labels = next(iter(train_loader))
    assert images.shape[0] == 2
    assert images.shape[1:] == (cfg.in_channels, cfg.image_size, cfg.image_size)
    assert labels.shape == (2,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long

    # id_test / ood_test splits should also honor the same (image, label) contract
    for loader in (id_loader, ood_loader):
        imgs, lbls = next(iter(loader))
        assert imgs.dtype == torch.float32
        assert lbls.dtype == torch.long
