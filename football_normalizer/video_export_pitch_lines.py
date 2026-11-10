"""Export SoccerNet's published DeepLabV3 calibration checkpoint for local inference."""
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    for name in ("mean", "std", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output path.")
    import numpy as np
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50

    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=29)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    # These settings are specified by SoccerNet's SegmentationNetwork initializer.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = .1

    class Export(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            self.register_buffer("mean", torch.tensor(np.load(args.mean), dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(np.load(args.std), dtype=torch.float32).view(1, 3, 1, 1))

        def forward(self, image):
            return self.model((image-self.mean)/self.std)["out"]

    torch.set_num_threads(4)
    exported = Export().eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(exported, torch.zeros(1, 3, 360, 640), str(args.output),
                      opset_version=17, dynamo=False, input_names=["image_bgr"], output_names=["line_logits"])
    print(args.output)


if __name__ == "__main__":
    main()
