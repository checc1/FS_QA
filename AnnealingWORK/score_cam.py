import matplotlib.cm as cm
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
from torchvision import models
from matplotlib import pyplot as plt
from feature_extraction_utils import grad_cam, FeatureExtractor, GradCAMpp
from compare_gurobi_qa import *
from sampling import single_sampling, evaluate_sample


class BaseCAM(object):
    """ Base class for Class activation mapping.

        : Args
            - **model_dict -** : Dict. Has format as dict(type='vgg', arch=torchvision.models.vgg16(pretrained=True),
            layer_name='features',input_size=(224, 224)).

    """

    def __init__(self, model: nn.Module, target_layer):
        #model_type = model_dict['type']
        #layer_name = model_dict['layer_name']
        self.model = model
        self.target_layer = target_layer
        #self.model_arch = model_dict['arch']
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            if torch.cuda.is_available():
                self.gradients['value'] = grad_output[0].cuda()
            else:
                self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            if torch.cuda.is_available():
                self.activations['value'] = output.cuda()
            else:
                self.activations['value'] = output
            return None

        self.fh = self.target_layer.register_forward_hook(forward_hook)
        self.bh = self.target_layer.register_full_backward_hook(backward_hook)

    def remove_hooks(self):
        """ Clean up hooks so subsequent extractors (like FeatureExtractor) can run freely. """
        self.fh.remove()
        self.bh.remove()

        #self.target_layer.register_forward_hook(forward_hook)
        #self.target_layer.register_backward_hook(backward_hook)

    def forward(self, input, class_idx=None, retain_graph=False):
        return None

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)



class ScoreCAM(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM

    """

    def __init__(self, model, target_layer):
        super().__init__(model, target_layer)

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        # predication on raw input
        logit = self.model(input).cuda()
        #logit = self.model_arch(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit, dim=-1)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            score = score
            #score = score.cuda()
            logit = logit
            #logit = logit.cuda()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            for i in range(k):

                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

                if saliency_map.max() == saliency_map.min():
                    continue

                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # how much increase if keeping the highlighted region
                # predication on masked input
                output = self.model(input * norm_saliency_map)
                output = F.softmax(output, dim=-1)
                score = output[0][predicted_class]

                score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None

        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(
            score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map.squeeze().cpu().numpy()

    def __call__(self, input, class_idx=None, retain_graph=False):
        #return self.forward(input, class_idx, retain_graph)
        res = self.forward(input, class_idx, retain_graph)
        self.remove_hooks()
        return res


def run_scorecam():
    class ReducedResNet18(nn.Module):
        def __init__(self, n_features: int, num_classes=10):
            self.n_features = n_features
            super().__init__()
            original_resnet = models.resnet18(weights="DEFAULT")
            self.features = nn.Sequential(
                original_resnet.conv1,
                original_resnet.bn1,
                original_resnet.relu,
                original_resnet.maxpool,
                original_resnet.layer1,
                nn.ReLU(inplace=False),
                original_resnet.layer2,
                nn.ReLU(inplace=False),
                original_resnet.layer3,
                nn.ReLU(inplace=False),
            )

            self.reduce = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=False),
                nn.Conv2d(256, self.n_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.n_features),
                nn.ReLU(inplace=False),
            )

            self.classifier = nn.Linear(self.n_features * 14 ** 2, num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.reduce(x)
            x = torch.flatten(x, 1)
            return self.classifier(x)

    index_class = 0
    idx = 18
    beta = 0.7
    features = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    filepath = os.path.join(
        'selected_fmaps',
        f'class{index_class}',
        f'Img_class_{index_class}_idx_{idx}_beta_{beta}_model{features}.npz'
    )

    if os.path.exists(filepath):
        input_tensor = torch.from_numpy(np.load(filepath)['input_tensor']).to(device).float().unsqueeze(0)

        model = ReducedResNet18(n_features=features).to(device)
        checkpoint_paths = {
            16: "checkpoint_ep17_fullResnetMod16.pth",
            24: "checkpoint_ep20_fullResnetMod24.pth",
            32: "checkpoint_ep6_fullResnetMod32.pth",
            512: "checkpoint_ep20_fullResnetMod512.pth"
        }

        if features not in checkpoint_paths:
            raise ValueError("Not implemented")

        checkpoint = torch.load(
            os.path.join(os.getcwd(), "checkpoints", checkpoint_paths[features]),
            map_location=device,
            weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        TARGET_LAYER = model.reduce[-1]

        resnet_model_scam = ScoreCAM(model=model, target_layer=TARGET_LAYER)
        scorecam_map = resnet_model_scam(input_tensor)

        extractor = FeatureExtractor()
        extractor.set_model(model)
        extractor.set_target_layer(TARGET_LAYER)
        extractor.set_input_tensor(input_tensor.squeeze(0))
        extractor.register_hooks()

        feat, gradients, pred = extractor.extract_features()

        cam_gc, _ = grad_cam(
            features=feat.cpu(),
            gradients=gradients.cpu(),
            subset_features=None,
        )

        gcpp = GradCAMpp(model=model, target_layer=TARGET_LAYER)
        cam_gcpp, _ = gcpp(input_tensor=input_tensor)

        cam_tensor = torch.from_numpy(cam_gc) if isinstance(cam_gc, np.ndarray) else cam_gc.cpu()
        campp_tensor = cam_gcpp

        if cam_tensor.dim() == 3:
            cam_tensor = cam_tensor.unsqueeze(0)
            campp_tensor = campp_tensor.unsqueeze(0)

        if cam_tensor.shape[1] > 1:
            cam_tensor = cam_tensor.mean(dim=1, keepdim=True)
            campp_tensor = campp_tensor.mean(dim=1, keepdim=True)

        cam_resized = torch.nn.functional.interpolate(
            cam_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze()

        campp_resized = torch.nn.functional.interpolate(
            campp_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
        ).squeeze()

        cam_min, cam_max = cam_resized.min(), cam_resized.max()
        if cam_max > cam_min:
            cam_resized = (cam_resized - cam_min) / (cam_max - cam_min)

        campp_min, campp_max = campp_resized.min(), campp_resized.max()
        if campp_max > campp_min:
            campp_resized = (campp_resized - campp_min) / (campp_max - campp_min)

        heatmap_inferno_gc = cm.coolwarm(cam_resized.numpy())[..., :3]
        heatmap_inferno_gcpp = cm.coolwarm(campp_resized.detach().clone().cpu().numpy())[..., :3]
        heatmap_inferno_sc = cm.coolwarm(scorecam_map)[..., :3]

        input_np = input_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if input_np.max() > 1.0:
            input_np = input_np / 255.0

        alpha = 0.5
        superimposed_gc = np.clip((1 - alpha) * input_np + alpha * heatmap_inferno_gc, 0, 1)
        superimposed_gcpp = np.clip((1 - alpha) * input_np + alpha * heatmap_inferno_gcpp, 0, 1)
        superimposed_sc = np.clip((1 - alpha) * input_np + alpha * heatmap_inferno_sc, 0, 1)

        _, bitstring_qa, _ = single_sampling(idx, index_class, 50, beta, features)
        superimposed_qa = evaluate_sample(input_tensor.squeeze(0), bitstring_qa, model, TARGET_LAYER)

        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].imshow(superimposed_gc)
        axs[1].imshow(superimposed_gcpp)
        axs[2].imshow(superimposed_sc)
        axs[3].imshow(superimposed_qa)
        axs[0].set_xlabel("GradCAM", fontsize=16)
        axs[1].set_xlabel("GradCAM++", fontsize=16)
        axs[2].set_xlabel("ScoreCAM", fontsize=16)
        axs[3].set_xlabel("QA-CAM", fontsize=16)

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.show()
    else:
        print(f"File missing: {filepath}")