"""Grad-CAM implemented directly with autograd hooks (no external library).

Ported from the coursework notebook: gradients of the predicted class score
w.r.t. the final conv layer's feature maps are global-average-pooled into
per-channel weights, combined with the activations, ReLU'd and normalized
into a heatmap. Modernized to use register_full_backward_hook and per-instance
state instead of module-level globals.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = target_layer or self._find_last_conv(model)
        self._handles = [
            layer.register_forward_hook(self._forward_hook),
            layer.register_full_backward_hook(self._backward_hook),
        ]

    @staticmethod
    def _find_last_conv(model: nn.Module) -> nn.Module:
        last = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last = module
        if last is None:
            raise ValueError("Model has no Conv2d layer")
        return last

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        for h in self._handles:
            h.remove()

    def heatmap(self, inp: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Compute the CAM for the model's predicted class of a 1xCxHxW input."""
        self.model.eval()
        out = self.model(inp)
        idx = out.argmax(dim=1).item()
        self.model.zero_grad()
        out[0, idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)
        return cam, idx


def overlay(img: Image.Image, cam: torch.Tensor, alpha: float = 0.5) -> Image.Image:
    heat = plt.cm.jet(cam.numpy())[..., :3]
    heat_img = Image.fromarray((heat * 255).astype("uint8")).resize(img.size)
    return Image.blend(img.convert("RGB"), heat_img, alpha=alpha)


def cam_grid(
    model: nn.Module,
    image_paths: list[str],
    preprocess,
    class_names: list[str],
    device,
    out_path: Path,
    title: str,
):
    """One row of Grad-CAM overlays for a list of images."""
    cam_engine = GradCAM(model)
    fig, axes = plt.subplots(1, len(image_paths), figsize=(2.6 * len(image_paths), 3))
    if len(image_paths) == 1:
        axes = [axes]
    for ax, path in zip(axes, image_paths):
        img = Image.open(path).convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(device)
        cam, idx = cam_engine.heatmap(inp)
        ax.imshow(overlay(img, cam))
        ax.set_title(f"pred: {class_names[idx]}", fontsize=9)
        ax.axis("off")
    cam_engine.remove()
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
