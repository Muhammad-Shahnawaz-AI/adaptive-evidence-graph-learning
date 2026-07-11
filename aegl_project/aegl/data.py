"""
Dataset utilities.

The proposal's benchmarking suite (Table 5.1) targets ImageNet-A/R,
iNaturalist 2024, MIMIC-IV / CMU-MOSEI, and PEMS -- all multi-GB external
downloads that this sandbox cannot fetch (no network access to dataset
hosts, only package registries). To keep the full AEGL pipeline runnable
end-to-end out of the box, `SyntheticVisionDataset` below generates a
lightweight proxy benchmark with the same *shape* of challenge:

  - `split="train"/"id_test"`: in-distribution samples -- each class is a
    fixed random spatial pattern (a stand-in for closed-world structure).
  - `split="ood_test"`: an out-of-distribution shift -- class patterns are
    rotated/rescaled and blended with extra noise, mimicking the
    long-tail / distribution-shift stress test used for ImageNet-A/R style
    evaluation.

To run on a real benchmark instead, replace `build_dataloaders` with a
`torchvision.datasets.ImageFolder` (or a custom iNaturalist/MIMIC loader)
that yields the same `(image_tensor, label)` contract -- no other code in
this project needs to change.
"""
import torch
from torch.utils.data import Dataset, DataLoader


class SyntheticVisionDataset(Dataset):
    def __init__(self, num_classes=10, image_size=32, in_channels=3,
                 samples_per_class=200, split="train", seed=0):
        self.num_classes = num_classes
        self.image_size = image_size
        self.in_channels = in_channels
        self.split = split

        gen = torch.Generator().manual_seed(seed)
        # One fixed "template" pattern per class -- the underlying closed-world structure.
        self.templates = torch.randn(
            num_classes, in_channels, image_size, image_size, generator=gen
        )

        n = samples_per_class * num_classes
        self.labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
        perm = torch.randperm(n, generator=gen)
        self.labels = self.labels[perm]

        noise_gen = torch.Generator().manual_seed(seed + 1)
        self.noise = torch.randn(n, in_channels, image_size, image_size, generator=noise_gen)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        y = self.labels[idx].item()
        template = self.templates[y]
        noise = self.noise[idx]

        if self.split == "ood_test":
            # Distribution shift: rotate channels, rescale, and inject heavier noise
            # to emulate an ImageNet-A/R style adversarial/natural shift.
            shifted = torch.flip(template, dims=[1]) * 0.6
            x = shifted + 0.9 * noise
        else:
            x = template + 0.35 * noise

        return x, y


def build_dataloaders(cfg, samples_per_class=200, batch_size=64, num_workers=0):
    train_ds = SyntheticVisionDataset(
        cfg.num_classes, cfg.image_size, cfg.in_channels,
        samples_per_class=samples_per_class, split="train", seed=0,
    )
    id_test_ds = SyntheticVisionDataset(
        cfg.num_classes, cfg.image_size, cfg.in_channels,
        samples_per_class=max(20, samples_per_class // 5), split="id_test", seed=100,
    )
    ood_test_ds = SyntheticVisionDataset(
        cfg.num_classes, cfg.image_size, cfg.in_channels,
        samples_per_class=max(20, samples_per_class // 5), split="ood_test", seed=200,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    id_loader = DataLoader(id_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    ood_loader = DataLoader(ood_test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, id_loader, ood_loader
