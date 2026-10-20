# Sources and artifact rights

This repository contains a new implementation inspired by [Nicholas Renotte's ImageClassification tutorial](https://github.com/nicknochnack/ImageClassification) and [YouTube walkthrough](https://www.youtube.com/watch?v=jztwpsIzEGc). It is not an official fork or an endorsement by the tutorial author.

The preserved reference dataset downloader pins upstream commit `ecbc2b219606c4153bdec47cd07f6ab00652c6d6`. Source photos are downloaded locally and are not included in this repository.

V2 uses [TorchVision's MobileNetV3 Small implementation and ImageNet weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html). Consult the [TorchVision license](https://github.com/pytorch/vision/blob/main/LICENSE) and the relevant pretrained-weight/data terms. The trained checkpoint is derived from those pretrained weights.

FER2013 happy/sad crops are obtained from [AutumnQiu/fer2013](https://huggingface.co/datasets/AutumnQiu/fer2013), pinned to revision `9c0267afc93fa4d4bfc05d7ff1108a04f72e8392`. FER2013 was introduced in the [2013 representation learning challenges](https://arxiv.org/abs/1307.0414).

The mirror does not provide an explicit image license. Neither the source images nor reconstructed image arrays are published here. Split IDs, decoded-pixel hashes, aggregate statistics, and model predictions are included for auditability. Check the relevant rights before redistributing images or using the model commercially.

No claim of state-of-the-art performance, identity-independent evaluation, or reliable inference of internal emotion is made.
