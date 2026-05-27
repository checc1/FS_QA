import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F


class FeatureExtractor:
    def __init__(self):
        self.model = None
        self.target_layer = None
        self.features = None
        self.input_tensor = None
        self.hook = None

    def set_model(self, model):
        self.model = model
        self.model.eval()

    def set_target_layer(self, target_layer):
        self.target_layer = target_layer
        self.features = None

    #def _save_features_hook(self, module, input, output):
    #    self.features = output
    #    self.features.retain_grad()

    def _save_features_hook(self, module, input, output):
        self.features = output
        if output.requires_grad:
            output.retain_grad()

    def register_hooks(self):
        self.remove_hooks()
        self.hook = self.target_layer.register_forward_hook(self._save_features_hook)

    def remove_hooks(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def set_input_tensor(self, input_tensor, label=None, see_picture=False):
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input tensor must be a torch.Tensor.")

        if input_tensor.dim() == 3:
            self.input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.dim() == 4:
            self.input_tensor = input_tensor
        else:
            raise ValueError("Input tensor must be 3D (C,H,W) or 4D (B,C,H,W).")

        self.input_tensor = self.input_tensor.detach().clone().requires_grad_(True)

        if see_picture:
            img = self.input_tensor[0].detach().cpu()
            if img.shape[0] in (1, 3):
                img_np = img.permute(1, 2, 0).numpy()
                plt.imshow(img_np.squeeze())
                plt.title(f"Label: {label}")
                plt.axis("off")
                plt.show()

    def extract_features(self):
        self.model.eval()

        if self.model is None:
            raise ValueError("Model must be set before extracting features.")
        if self.target_layer is None:
            raise ValueError("Target layer must be set before extracting features.")
        if self.input_tensor is None:
            raise ValueError("Input tensor must be set before extracting features.")

        device = next(self.model.parameters()).device
        x = self.input_tensor.to(device)

        self.model.zero_grad(set_to_none=True)
        output = self.model(x)
        pred_class = output.argmax(dim=1).item()

        class_score = output[0, pred_class]
        class_score.backward()

        if self.features is None:
            raise RuntimeError("Forward hook did not capture features.")
        if self.features.grad is None:
            raise RuntimeError("Gradients were not retained on the captured features.")

        gradients = self.features.grad

        return self.features.detach().cpu(), gradients.detach().cpu(), pred_class