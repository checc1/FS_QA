import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from model import ReducedResNet18
from feature_extraction_utils import grad_cam, FeatureExtractor, GradCAMpp
from sampling import single_sampling
from score_cam import ScoreCAM


class AvgDrop():
    def __init__(self,
                 method: str,
                 model,
                 target_layer: int,
                 index_class: int,
                 n_features: int,
                 device: str) -> None:

        self.method = method
        self.model = model
        self.target_layer = target_layer
        self.index_class = index_class
        self.n_features = n_features
        self.device = device

    def __load__(self) -> tuple:
        if self.device in ["cpu", "cuda"]:
            self.model = ReducedResNet18(n_features=self.n_features).to(torch.device(self.device))
        else:
            raise ValueError("Device not supported")

        checkpoint_paths = {
            16: "checkpoint_ep17_fullResnetMod16.pth",
            24: "checkpoint_ep20_fullResnetMod24.pth",
            32: "checkpoint_ep6_fullResnetMod32.pth",
            512: "checkpoint_ep20_fullResnetMod512.pth"
        }
        if self.n_features not in checkpoint_paths:
            raise ValueError("Not implemented")

        checkpoint = torch.load(
            os.path.join(os.getcwd(), "checkpoints", checkpoint_paths[self.n_features]),
            map_location=self.device,
            weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        target_layer_selected = self.model.reduce[self.target_layer]
        return self.model, target_layer_selected

    def __extract__(self, beta: float | int, threshold: float, mode: str = "single", seed=8888) -> tuple:

        np.random.seed(seed)

        if self.method == "qa":
            filepath = os.path.join(
                'selected_fmaps',
                f'class{index_class}',
            )

            tau = 50
            if mode == "single":
                input_index = np.random.randint(low=0, high=20, size=1)[0]
                item = os.path.join(filepath, f'Img_class_{self.index_class}_idx_{input_index}_beta_{beta}_model{self.n_features}.npz')
                if os.path.exists(item):
                    input_tensor = torch.from_numpy(np.load(item)['input_tensor']).to(
                        torch.device(self.device)).float().unsqueeze(0)
                else:
                    raise ValueError(f"There is no input image for index {input_index} and class {self.index_class}")

                model_to_use, target_layer_to_use = self.__load__()

                extractor = FeatureExtractor()
                extractor.set_model(model_to_use)
                extractor.set_target_layer(target_layer_to_use)
                extractor.set_input_tensor(input_tensor.squeeze(0))
                extractor.register_hooks()

                feat, gradients, pred = extractor.extract_features()
                extractor.remove_hooks()

                _, bitstring, _ = single_sampling(input_index, self.index_class, tau, beta, self.n_features)

                cam_gc, _ = grad_cam(
                    features=feat.cpu(),
                    gradients=gradients.cpu(),
                    subset_features=bitstring,
                )

                cam_tensor = torch.from_numpy(cam_gc) if isinstance(cam_gc, np.ndarray) else cam_gc.cpu()
                if cam_tensor.dim() == 3:
                    cam_tensor = cam_tensor.unsqueeze(0)
                if cam_tensor.shape[1] > 1:
                    cam_tensor = cam_tensor.mean(dim=1, keepdim=True)

                cam_resized = torch.nn.functional.interpolate(
                    cam_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze()

                cam_min, cam_max = cam_resized.min(), cam_resized.max()
                cam_normalized = (cam_resized - cam_min) / (cam_max - cam_min) if cam_max > cam_min else torch.zeros_like(
                    cam_resized)

                binary_mask = (cam_normalized >= threshold).float().to(self.device)
                masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                plt.imshow(masked_image.detach().cpu().permute(1, 2, 0).numpy())
                plt.show()

                return input_tensor, masked_image, input_index

            elif mode == "multiple":
                input_tensor_list, masked_tensor_list = [], []
                valid_indices = []
                model_to_use, target_layer_to_use = self.__load__()

                for i in range(20):
                    item = os.path.join(
                        filepath,
                        f'Img_class_{index_class}_idx_{i}_beta_{beta}_model{features}.npz'
                    )
                    try:
                        if os.path.exists(item):
                            npz_file = np.load(item)
                            if 'input_tensor' not in npz_file:
                                raise KeyError(f"'input_tensor' key missing in {item}")
                            input_tensor = torch.from_numpy(npz_file['input_tensor']).to(
                                torch.device(self.device)).float().unsqueeze(0)
                        else:
                            raise FileNotFoundError(f"There is no input image for index {i} and class {self.index_class}")
                    except (FileNotFoundError, KeyError):
                        continue

                    extractor = FeatureExtractor()
                    extractor.set_model(model_to_use)
                    extractor.set_target_layer(target_layer_to_use)
                    extractor.set_input_tensor(input_tensor.squeeze(0))
                    extractor.register_hooks()

                    feat, gradients, pred = extractor.extract_features()
                    extractor.remove_hooks()

                    _, bitstring, _ = single_sampling(i, self.index_class, tau, beta, self.n_features)

                    cam_gc, _ = grad_cam(
                        features=feat.cpu(),
                        gradients=gradients.cpu(),
                        subset_features=bitstring,
                    )

                    cam_tensor = torch.from_numpy(cam_gc) if isinstance(cam_gc, np.ndarray) else cam_gc.cpu()
                    if cam_tensor.dim() == 3:
                        cam_tensor = cam_tensor.unsqueeze(0)
                    if cam_tensor.shape[1] > 1:
                        cam_tensor = cam_tensor.mean(dim=1, keepdim=True)

                    cam_resized = torch.nn.functional.interpolate(
                        cam_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
                    ).squeeze()

                    cam_min, cam_max = cam_resized.min(), cam_resized.max()
                    cam_normalized = (cam_resized - cam_min) / (
                                cam_max - cam_min) if cam_max > cam_min else torch.zeros_like(cam_resized)

                    binary_mask = (cam_normalized >= threshold).float().to(self.device)
                    masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                    input_tensor_list.append(input_tensor)
                    masked_tensor_list.append(masked_image)
                    valid_indices.append(i)

                if len(input_tensor_list) == 0:
                    raise ValueError(f"No valid images were successfully loaded for class {self.index_class}")

                return input_tensor_list, masked_tensor_list, valid_indices
            else:
                raise ValueError("Mode not implemented yet")

        elif self.method == "sa":
            filepath = os.path.join(
                'sa_results',
                f'F{self.n_features}'
            )

            if mode == "single":
                input_index = np.random.randint(low=0, high=20, size=1)[0]
                item = os.path.join(filepath, f'optimal_solutions_{self.index_class}_idx_{input_index}_beta_{beta}.npz')
                if os.path.exists(item):
                    input_tensor = torch.from_numpy(np.load(item)['input_tensor']).to(
                        torch.device(self.device)).float().unsqueeze(0)
                else:
                    raise ValueError(f"There is no input image for index {input_index} and class {self.index_class}")

                model_to_use, target_layer_to_use = self.__load__()

                extractor = FeatureExtractor()
                extractor.set_model(model_to_use)
                extractor.set_target_layer(target_layer_to_use)
                extractor.set_input_tensor(input_tensor.squeeze(0))
                extractor.register_hooks()

                feat, gradients, pred = extractor.extract_features()
                extractor.remove_hooks()

                bitstring_sa = np.load(
                    os.path.join(filepath, f'optimal_solutions_{self.index_class}_idx_{input_index}_beta_{beta}.npz')
                )['solutions']

                cam_gc, _ = grad_cam(
                    features=feat.cpu(),
                    gradients=gradients.cpu(),
                    subset_features=bitstring_sa,
                )

                cam_tensor = torch.from_numpy(cam_gc) if isinstance(cam_gc, np.ndarray) else cam_gc.cpu()
                if cam_tensor.dim() == 3:
                    cam_tensor = cam_tensor.unsqueeze(0)
                if cam_tensor.shape[1] > 1:
                    cam_tensor = cam_tensor.mean(dim=1, keepdim=True)

                cam_resized = torch.nn.functional.interpolate(
                    cam_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze()

                cam_min, cam_max = cam_resized.min(), cam_resized.max()
                cam_normalized = (cam_resized - cam_min) / (
                            cam_max - cam_min) if cam_max > cam_min else torch.zeros_like(
                    cam_resized)

                binary_mask = (cam_normalized >= threshold).float().to(self.device)
                masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                return input_tensor, masked_image, input_index

            elif mode == "multiple":
                input_tensor_list, masked_tensor_list = [], []
                valid_indices = []
                model_to_use, target_layer_to_use = self.__load__()

                for i in range(20):
                    item = os.path.join(
                        filepath,
                        f'optimal_solutions_{self.index_class}_idx_{i}_beta_{beta}.npz'
                    )
                    try:
                        if os.path.exists(item):
                            npz_file = np.load(item)
                            if 'input_tensor' not in npz_file:
                                raise KeyError(f"'input_tensor' key missing in {item}")
                            input_tensor = torch.from_numpy(npz_file['input_tensor']).to(
                                torch.device(self.device)).float().unsqueeze(0)
                        else:
                            raise FileNotFoundError(
                                f"There is no input image for index {i} and class {self.index_class}")
                    except (FileNotFoundError, KeyError):
                        continue

                    extractor = FeatureExtractor()
                    extractor.set_model(model_to_use)
                    extractor.set_target_layer(target_layer_to_use)
                    extractor.set_input_tensor(input_tensor.squeeze(0))
                    extractor.register_hooks()

                    feat, gradients, pred = extractor.extract_features()
                    extractor.remove_hooks()

                    bitstring_sa = np.load(
                        os.path.join(filepath,
                                     f'optimal_solutions_{self.index_class}_idx_{i}_beta_{beta}.npz')
                    )['solutions']

                    cam_gc, _ = grad_cam(
                        features=feat.cpu(),
                        gradients=gradients.cpu(),
                        subset_features=bitstring_sa,
                    )

                    cam_tensor = torch.from_numpy(cam_gc) if isinstance(cam_gc, np.ndarray) else cam_gc.cpu()
                    if cam_tensor.dim() == 3:
                        cam_tensor = cam_tensor.unsqueeze(0)
                    if cam_tensor.shape[1] > 1:
                        cam_tensor = cam_tensor.mean(dim=1, keepdim=True)

                    cam_resized = torch.nn.functional.interpolate(
                        cam_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
                    ).squeeze()

                    cam_min, cam_max = cam_resized.min(), cam_resized.max()
                    cam_normalized = (cam_resized - cam_min) / (
                            cam_max - cam_min) if cam_max > cam_min else torch.zeros_like(cam_resized)

                    binary_mask = (cam_normalized >= threshold).float().to(self.device)
                    masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                    input_tensor_list.append(input_tensor)
                    masked_tensor_list.append(masked_image)
                    valid_indices.append(i)

                if len(input_tensor_list) == 0:
                    raise ValueError(f"No valid images were successfully loaded for class {self.index_class}")

                return input_tensor_list, masked_tensor_list, valid_indices
            else:
                raise ValueError("Mode not implemented yet")

        elif self.method == "gurobi":
            filepath = os.path.join(
                'gurobi_results',
                f'F{self.n_features}'
            )

            if mode == "single":
                input_index = np.random.randint(low=0, high=20, size=1)[0]
                item = os.path.join(filepath,
                                    f'optimal_solutions_{self.index_class}_idx_{input_index}_beta_{beta}.npz')
                if os.path.exists(item):
                    input_tensor = torch.from_numpy(np.load(item)['input_tensor']).to(
                        torch.device(self.device)).float().unsqueeze(0)
                else:
                    raise ValueError(
                        f"There is no input image for index {input_index} and class {self.index_class}")

                model_to_use, target_layer_to_use = self.__load__()

                extractor = FeatureExtractor()
                extractor.set_model(model_to_use)
                extractor.set_target_layer(target_layer_to_use)
                extractor.set_input_tensor(input_tensor.squeeze(0))
                extractor.register_hooks()

                feat, gradients, pred = extractor.extract_features()
                extractor.remove_hooks()

                bitstring_gb = np.load(
                    os.path.join(filepath,
                                 f'optimal_solutions_{self.index_class}_idx_{input_index}_beta_{beta}.npz')
                )['solutions']

                cam_gc, _ = grad_cam(
                    features=feat.cpu(),
                    gradients=gradients.cpu(),
                    subset_features=bitstring_gb,
                )

                cam_tensor = torch.from_numpy(cam_gc) if isinstance(cam_gc, np.ndarray) else cam_gc.cpu()
                if cam_tensor.dim() == 3:
                    cam_tensor = cam_tensor.unsqueeze(0)
                if cam_tensor.shape[1] > 1:
                    cam_tensor = cam_tensor.mean(dim=1, keepdim=True)

                cam_resized = torch.nn.functional.interpolate(
                    cam_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze()

                cam_min, cam_max = cam_resized.min(), cam_resized.max()
                cam_normalized = (cam_resized - cam_min) / (
                        cam_max - cam_min) if cam_max > cam_min else torch.zeros_like(
                    cam_resized)

                binary_mask = (cam_normalized >= threshold).float().to(self.device)
                masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                plt.imshow(masked_image.detach().cpu().permute(1, 2, 0).numpy())
                plt.show()

                return input_tensor, masked_image, input_index

            elif mode == "multiple":
                input_tensor_list, masked_tensor_list = [], []
                valid_indices = []
                model_to_use, target_layer_to_use = self.__load__()

                for i in range(20):
                    item = os.path.join(
                        filepath,
                        f'optimal_solutions_{self.index_class}_idx_{i}_beta_{beta}.npz'
                    )
                    try:
                        if os.path.exists(item):
                            npz_file = np.load(item)
                            if 'input_tensor' not in npz_file:
                                raise KeyError(f"'input_tensor' key missing in {item}")
                            input_tensor = torch.from_numpy(npz_file['input_tensor']).to(
                                torch.device(self.device)).float().unsqueeze(0)
                        else:
                            raise FileNotFoundError(
                                f"There is no input image for index {i} and class {self.index_class}")
                    except (FileNotFoundError, KeyError):
                        continue

                    extractor = FeatureExtractor()
                    extractor.set_model(model_to_use)
                    extractor.set_target_layer(target_layer_to_use)
                    extractor.set_input_tensor(input_tensor.squeeze(0))
                    extractor.register_hooks()

                    feat, gradients, pred = extractor.extract_features()
                    extractor.remove_hooks()

                    bitstring_gb = np.load(
                        os.path.join(filepath,
                                     f'optimal_solutions_{self.index_class}_idx_{i}_beta_{beta}.npz')
                    )['solutions']

                    cam_gc, _ = grad_cam(
                        features=feat.cpu(),
                        gradients=gradients.cpu(),
                        subset_features=bitstring_gb,
                    )

                    cam_tensor = torch.from_numpy(cam_gc) if isinstance(cam_gc, np.ndarray) else cam_gc.cpu()
                    if cam_tensor.dim() == 3:
                        cam_tensor = cam_tensor.unsqueeze(0)
                    if cam_tensor.shape[1] > 1:
                        cam_tensor = cam_tensor.mean(dim=1, keepdim=True)

                    cam_resized = torch.nn.functional.interpolate(
                        cam_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
                    ).squeeze()

                    cam_min, cam_max = cam_resized.min(), cam_resized.max()
                    cam_normalized = (cam_resized - cam_min) / (
                            cam_max - cam_min) if cam_max > cam_min else torch.zeros_like(cam_resized)

                    binary_mask = (cam_normalized >= threshold).float().to(self.device)
                    masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                    input_tensor_list.append(input_tensor)
                    masked_tensor_list.append(masked_image)
                    valid_indices.append(i)

                if len(input_tensor_list) == 0:
                    raise ValueError(f"No valid images were successfully loaded for class {self.index_class}")

                return input_tensor_list, masked_tensor_list, valid_indices
            else:
                raise ValueError("Mode not implemented yet")

        elif self.method == "scorecam":
            filepath = os.path.join(
                'selected_fmaps',
                f'class{self.index_class}',
            )

            if mode == "single":
                input_index = np.random.randint(low=0, high=20, size=1)[0]
                item = os.path.join(filepath,
                                    f'Img_class_{self.index_class}_idx_{input_index}_beta_{beta}_model{self.n_features}.npz')
                if os.path.exists(item):
                    input_tensor = torch.from_numpy(np.load(item)['input_tensor']).to(
                        torch.device(self.device)).float()
                    if input_tensor.dim() == 3:
                        input_tensor = input_tensor.unsqueeze(0)
                    elif input_tensor.dim() == 5:
                        input_tensor = input_tensor.squeeze(0).squeeze(0)

                else:
                    raise ValueError(
                        f"There is no input image for index {input_index} and class {self.index_class}")

                model_to_use, target_layer_to_use = self.__load__()

                resnet_model_scam = ScoreCAM(model=model_to_use, target_layer=target_layer_to_use)
                scorecam_map = resnet_model_scam(input_tensor)

                if isinstance(scorecam_map, np.ndarray):
                    scorecam_resized = torch.from_numpy(scorecam_map).float().to(self.device)
                else:
                    scorecam_resized = scorecam_map.float().to(self.device)

                if scorecam_resized.dim() != 2 or scorecam_resized.shape != input_tensor.shape[-2:]:
                    while scorecam_resized.dim() < 4:
                        scorecam_resized = scorecam_resized.unsqueeze(0)

                    scorecam_resized = torch.nn.functional.interpolate(
                        scorecam_resized,
                        size=input_tensor.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                scorecam_min, scorecam_max = scorecam_resized.min(), scorecam_resized.max()
                scorecam_normalized = (scorecam_resized - scorecam_min) / (
                        scorecam_max - scorecam_min) if scorecam_max > scorecam_min else torch.zeros_like(
                    scorecam_resized)

                binary_mask = (scorecam_normalized >= threshold).float().to(self.device)
                masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                #plt.imshow(masked_image.detach().cpu().permute(1, 2, 0).numpy())
                #plt.show()

                return input_tensor, masked_image, input_index

            elif mode == "multiple":
                input_tensor_list, masked_tensor_list = [], []
                valid_indices = []
                model_to_use, target_layer_to_use = self.__load__()

                for i in range(20):
                    item = os.path.join(
                        filepath,
                        f'Img_class_{self.index_class}_idx_{i}_beta_{beta}_model{self.n_features}.npz'
                    )
                    try:
                        if os.path.exists(item):
                            npz_file = np.load(item)
                            if 'input_tensor' not in npz_file:
                                raise KeyError(f"'input_tensor' key missing in {item}")
                            input_tensor = torch.from_numpy(npz_file['input_tensor']).to(
                                torch.device(self.device)).float().unsqueeze(0)
                        else:
                            raise FileNotFoundError(
                                f"There is no input image for index {i} and class {self.index_class}")
                    except (FileNotFoundError, KeyError):
                        continue

                    resnet_model_scam = ScoreCAM(model=model_to_use, target_layer=target_layer_to_use)
                    scorecam_map = resnet_model_scam(input_tensor)

                    if isinstance(scorecam_map, np.ndarray):
                        scorecam_resized = torch.from_numpy(scorecam_map).float().to(self.device)
                    else:
                        scorecam_resized = scorecam_map.float().to(self.device)

                    if scorecam_resized.dim() != 2 or scorecam_resized.shape != input_tensor.shape[-2:]:
                        while scorecam_resized.dim() < 4:
                            scorecam_resized = scorecam_resized.unsqueeze(0)

                        scorecam_resized = torch.nn.functional.interpolate(
                            scorecam_resized,
                            size=input_tensor.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()

                    scorecam_min, scorecam_max = scorecam_resized.min(), scorecam_resized.max()
                    scorecam_normalized = (scorecam_resized - scorecam_min) / (
                            scorecam_max - scorecam_min) if scorecam_max > scorecam_min else torch.zeros_like(
                        scorecam_resized)

                    binary_mask = (scorecam_normalized >= threshold).float().to(self.device)
                    masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                    input_tensor_list.append(input_tensor)
                    masked_tensor_list.append(masked_image)
                    valid_indices.append(i)

                if len(input_tensor_list) == 0:
                    raise ValueError(f"No valid images were successfully loaded for class {self.index_class}")

                return input_tensor_list, masked_tensor_list, valid_indices
            else:
                raise ValueError("Mode not implemented yet")

        elif self.method == "gradcam":
            filepath = os.path.join(
                'selected_fmaps',
                f'class{self.index_class}',
            )

            if mode == "single":
                input_index = np.random.randint(low=0, high=20, size=1)[0]
                item = os.path.join(filepath,
                                    f'Img_class_{self.index_class}_idx_{input_index}_beta_{beta}_model{self.n_features}.npz')
                if os.path.exists(item):
                    input_tensor = torch.from_numpy(np.load(item)['input_tensor']).to(
                        torch.device(self.device)).float().unsqueeze(0)
                else:
                    raise ValueError(
                        f"There is no input image for index {input_index} and class {self.index_class}")

                model_to_use, target_layer_to_use = self.__load__()

                extractor = FeatureExtractor()
                extractor.set_model(model_to_use)
                extractor.set_target_layer(target_layer_to_use)
                extractor.set_input_tensor(input_tensor.squeeze(0))
                extractor.register_hooks()

                feat, gradients, pred = extractor.extract_features()

                cam_gc, _ = grad_cam(
                    features=feat.cpu(),
                    gradients=gradients.cpu(),
                    subset_features=None,
                )

                gcam_tensor = torch.from_numpy(cam_gc) if isinstance(cam_gc, np.ndarray) else cam_gc.cpu()
                if gcam_tensor.dim() == 3:
                    gcam_tensor = gcam_tensor.unsqueeze(0)
                if gcam_tensor.shape[1] > 1:
                    gcam_tensor = gcam_tensor.mean(dim=1, keepdim=True)

                gcam_resized = torch.nn.functional.interpolate(
                    gcam_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
                ).squeeze()

                gcam_min, gcam_max = gcam_resized.min(), gcam_resized.max()
                gcam_normalized = (gcam_resized - gcam_min) / (
                        gcam_max - gcam_min) if gcam_max > gcam_min else torch.zeros_like(
                    gcam_resized)

                binary_mask = (gcam_normalized >= threshold).float().to(self.device)
                masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                plt.imshow(masked_image.detach().cpu().permute(1,2,0).numpy())
                plt.show()

                return input_tensor, masked_image, input_index

            elif mode == "multiple":
                input_tensor_list, masked_tensor_list = [], []
                valid_indices = []
                model_to_use, target_layer_to_use = self.__load__()

                for i in range(20):
                    item = os.path.join(
                        filepath,
                        f'Img_class_{self.index_class}_idx_{i}_beta_{beta}_model{self.n_features}.npz'
                    )
                    try:
                        if os.path.exists(item):
                            npz_file = np.load(item)
                            if 'input_tensor' not in npz_file:
                                raise KeyError(f"'input_tensor' key missing in {item}")
                            input_tensor = torch.from_numpy(npz_file['input_tensor']).to(
                                torch.device(self.device)).float().unsqueeze(0)
                        else:
                            raise FileNotFoundError(
                                f"There is no input image for index {i} and class {self.index_class}")
                    except (FileNotFoundError, KeyError):
                        continue

                    extractor = FeatureExtractor()
                    extractor.set_model(model_to_use)
                    extractor.set_target_layer(target_layer_to_use)
                    extractor.set_input_tensor(input_tensor.squeeze(0))
                    extractor.register_hooks()

                    feat, gradients, pred = extractor.extract_features()

                    cam_gc, _ = grad_cam(
                        features=feat.cpu(),
                        gradients=gradients.cpu(),
                        subset_features=None,
                    )

                    gcam_tensor = torch.from_numpy(cam_gc) if isinstance(cam_gc, np.ndarray) else cam_gc.cpu()
                    if gcam_tensor.dim() == 3:
                        gcam_tensor = gcam_tensor.unsqueeze(0)
                    if gcam_tensor.shape[1] > 1:
                        gcam_tensor = gcam_tensor.mean(dim=1, keepdim=True)

                    gcam_resized = torch.nn.functional.interpolate(
                        gcam_tensor, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False
                    ).squeeze()

                    gcam_min, gcam_max = gcam_resized.min(), gcam_resized.max()
                    gcam_normalized = (gcam_resized - gcam_min) / (
                            gcam_max - gcam_min) if gcam_max > gcam_min else torch.zeros_like(
                        gcam_resized)

                    binary_mask = (gcam_normalized >= threshold).float().to(self.device)
                    masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                    input_tensor_list.append(input_tensor)
                    masked_tensor_list.append(masked_image)
                    valid_indices.append(i)

                if len(input_tensor_list) == 0:
                    raise ValueError(f"No valid images were successfully loaded for class {self.index_class}")

                return input_tensor_list, masked_tensor_list, valid_indices
            else:
                raise ValueError("Mode not implemented yet")

        elif self.method == "gradcampp":
            filepath = os.path.join(
                'selected_fmaps',
                f'class{self.index_class}',
            )

            if mode == "single":
                input_index = np.random.randint(low=0, high=20, size=1)[0]
                item = os.path.join(filepath,
                                    f'Img_class_{self.index_class}_idx_{input_index}_beta_{beta}_model{self.n_features}.npz')
                if os.path.exists(item):
                    input_tensor = torch.from_numpy(np.load(item)['input_tensor']).to(
                        torch.device(self.device)).float().unsqueeze(0)
                else:
                    raise ValueError(
                        f"There is no input image for index {input_index} and class {self.index_class}")

                model_to_use, target_layer_to_use = self.__load__()

                gcpp = GradCAMpp(model=model_to_use, target_layer=target_layer_to_use)
                cam_gcpp, _ = gcpp(input_tensor=input_tensor)


                if isinstance(cam_gcpp, np.ndarray):
                    gcampp_resized = torch.from_numpy(cam_gcpp).float().to(self.device)
                else:
                    gcampp_resized = cam_gcpp.float().to(self.device)

                if gcampp_resized.dim() != 2 or gcampp_resized.shape != input_tensor.shape[-2:]:
                    while gcampp_resized.dim() < 4:
                        gcampp_resized = gcampp_resized.unsqueeze(0)

                    gcampp_resized = torch.nn.functional.interpolate(
                        gcampp_resized,
                        size=input_tensor.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()


                gcampp_min, gcampp_max = gcampp_resized.min(), gcampp_resized.max()
                gcampp_normalized = (gcampp_resized - gcampp_min) / (
                        gcampp_max - gcampp_min) if gcampp_max > gcampp_min else torch.zeros_like(
                    gcampp_resized)

                binary_mask = (gcampp_normalized >= threshold).float().to(self.device)
                masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                plt.imshow(masked_image.detach().cpu().permute(1, 2, 0).numpy())
                plt.show()

                return input_tensor, masked_image, input_index

            elif mode == "multiple":
                input_tensor_list, masked_tensor_list = [], []
                valid_indices = []
                model_to_use, target_layer_to_use = self.__load__()

                for i in range(20):
                    item = os.path.join(
                        filepath,
                        f'Img_class_{self.index_class}_idx_{i}_beta_{beta}_model{self.n_features}.npz'
                    )
                    try:
                        if os.path.exists(item):
                            npz_file = np.load(item)
                            if 'input_tensor' not in npz_file:
                                raise KeyError(f"'input_tensor' key missing in {item}")
                            input_tensor = torch.from_numpy(npz_file['input_tensor']).to(
                                torch.device(self.device)).float().unsqueeze(0)
                        else:
                            raise FileNotFoundError(
                                f"There is no input image for index {i} and class {self.index_class}")
                    except (FileNotFoundError, KeyError):
                        continue

                    gcpp = GradCAMpp(model=model_to_use, target_layer=target_layer_to_use)
                    cam_gcpp, _ = gcpp(input_tensor=input_tensor)

                    if isinstance(cam_gcpp, np.ndarray):
                        gcampp_resized = torch.from_numpy(cam_gcpp).float().to(self.device)
                    else:
                        gcampp_resized = cam_gcpp.float().to(self.device)

                    if gcampp_resized.dim() != 2 or gcampp_resized.shape != input_tensor.shape[-2:]:
                        while gcampp_resized.dim() < 4:
                            gcampp_resized = gcampp_resized.unsqueeze(0)

                        gcampp_resized = torch.nn.functional.interpolate(
                            gcampp_resized,
                            size=input_tensor.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()

                    gcampp_min, gcampp_max = gcampp_resized.min(), gcampp_resized.max()
                    gcampp_normalized = (gcampp_resized - gcampp_min) / (
                            gcampp_max - gcampp_min) if gcampp_max > gcampp_min else torch.zeros_like(
                        gcampp_resized)

                    binary_mask = (gcampp_normalized >= threshold).float().to(self.device)
                    masked_image = input_tensor.squeeze(0) * binary_mask.unsqueeze(0)

                    input_tensor_list.append(input_tensor)
                    masked_tensor_list.append(masked_image)
                    valid_indices.append(i)

                if len(input_tensor_list) == 0:
                    raise ValueError(f"No valid images were successfully loaded for class {self.index_class}")

                return input_tensor_list, masked_tensor_list, valid_indices
            else:
                raise ValueError("Mode not implemented yet")

    def compute(self, beta: float | int, threshold: float, mode: str = "single"):
        if mode == "single":
            input_tensor, masked_image, idx = self.__extract__(beta, threshold, mode)
            masked_image = masked_image.unsqueeze(0)

            model_to_use, _ = self.__load__()
            model_to_use.eval()

            with torch.no_grad():
                logits_original = model_to_use(input_tensor)
                logits_masked = model_to_use(masked_image)
                probs_original = torch.softmax(logits_original, dim=-1).squeeze(0)
                probs_masked = torch.softmax(logits_masked, dim=-1).squeeze(0)

            conf_original = probs_original[self.index_class].item()
            conf_masked = probs_masked[self.index_class].item()

            avg_drop = max(0.0, (conf_original - conf_masked) / max(conf_original, 1e-7)) * 100
            increase_in_confidence = 1.0 if conf_masked > conf_original else 0.0

            pred_original = torch.argmax(probs_original).item()
            pred_masked = torch.argmax(probs_masked).item()

            return {
                "target_class": self.index_class,
                "image_index": int(idx),
                "original_prediction_class": pred_original,
                "masked_prediction_class": pred_masked,
                "original_confidence": float(np.round(conf_original, 4)),
                "masked_confidence": float(np.round(conf_masked, 4)),
                "average_drop_percent": float(np.round(avg_drop, 4)),
                "confidence_increased": bool(increase_in_confidence)
            }

        elif mode == "multiple":
            model_to_use, _ = self.__load__()
            model_to_use.eval()
            input_tensor_list, masked_image_list, idx_list = self.__extract__(beta, threshold, mode)

            results_table = {
                "sample_idx": [],
                "target_class": [],
                "original_pred_class": [],
                "masked_pred_class": [],
                "original_confidence": [],
                "masked_confidence": [],
                "average_drop_percent": [],
                "confidence_increased": []
            }

            for file_t, file_m, img_idx in zip(input_tensor_list, masked_image_list, idx_list):
                file_m = file_m.unsqueeze(0)

                with torch.no_grad():
                    logits_original = model_to_use(file_t)
                    logits_masked = model_to_use(file_m)

                    probs_original = torch.softmax(logits_original, dim=-1).squeeze(0)
                    probs_masked = torch.softmax(logits_masked, dim=-1).squeeze(0)

                conf_original = probs_original[self.index_class].item()
                conf_masked = probs_masked[self.index_class].item()

                avg_drop = max(0.0, (conf_original - conf_masked) / max(conf_original, 1e-7)) * 100
                increase_in_confidence = 1.0 if conf_masked > conf_original else 0.0

                pred_original = torch.argmax(probs_original).item()
                pred_masked = torch.argmax(probs_masked).item()

                results_table["sample_idx"].append(int(img_idx))
                results_table["target_class"].append(self.index_class)
                results_table["original_pred_class"].append(pred_original)
                results_table["masked_pred_class"].append(pred_masked)
                results_table["original_confidence"].append(float(np.round(conf_original, 4)))
                results_table["masked_confidence"].append(float(np.round(conf_masked, 4)))
                results_table["average_drop_percent"].append(float(np.round(avg_drop, 4)))
                results_table["confidence_increased"].append(bool(increase_in_confidence))

            return results_table
        else:
            raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    index_class = 0
    beta = 0.7
    features = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ReducedResNet18(n_features=features).to(device)
    chosen_method = "qa"
    drop = AvgDrop(chosen_method, model, -1, 0, features, "cuda")
    result = drop.compute(beta, 0.5, "multiple")

    print(f"Result with {chosen_method}:\n", result)