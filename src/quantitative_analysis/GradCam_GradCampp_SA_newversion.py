import os
import re
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
from PIL import Image
from tqdm import tqdm
from qutip_class import *
from sampling import sample_minimum_energy, bitstring_basis_msb
from src.feature_extraction_utils2 import FeatureExtractor
import torch.nn as nn
from torchvision import models
warnings.filterwarnings('ignore')

DEVICE = torch.device("cpu")

F = 512
beta = 0.99

CLASS_NAMES = [
    'Airplane', 'Bird', 'Car', 'Cat', 'Deer',
    'Dog', 'Horse', 'Monkey', 'Ship', 'Truck'
]

NUM_CLASSES = len(CLASS_NAMES)

class OriginalResNet18(nn.Module):
    def __init__(self, num_classes=10, reduced_features=F):
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
            original_resnet.layer4,
            nn.ReLU(inplace=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.classifier = nn.Linear(reduced_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ReducedResNet18(nn.Module):
    def __init__(self, num_classes=10, reduced_features=F):
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
            nn.Conv2d(256, reduced_features, kernel_size=3, padding=1),
            nn.BatchNorm2d(reduced_features),
            nn.ReLU(inplace=False),
        )

        self.classifier = nn.Linear(reduced_features * 14**2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)



device = torch.device("cpu")
#model = ReducedResNet18(num_classes=10, reduced_features=F)
model = OriginalResNet18(num_classes=10, reduced_features=F)
model.to(device)
checkpoint = torch.load(
    #os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep17_fullResnetMod16.pth"),
    #os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep20_fullResnetMod24.pth"),
    os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep20_fullResnetMod512.pth"),
    map_location=device
)

test_set = torch.load(os.path.join(os.getcwd(), "data", "data", "test_dataset.pth"), map_location=device, weights_only=False)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
TARGET_LAYER = model.features[-1]
#TARGET_LAYER = model.reduce[-1]

class GradCAMpp:
    def __init__(self, model, target_layer, verbose=False):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.fwd_hook = target_layer.register_forward_hook(forward_hook)
        self.bwd_hook = target_layer.register_full_backward_hook(backward_hook)

        if verbose:
            print("✓ GradCAM++ initialized")

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        score = logits[0, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        b, k, u, v = gradients.shape

        # GradCAM++ weights
        grad_sq = gradients.pow(2)
        grad_cube = gradients.pow(3)

        denom = 2 * grad_sq + (
            activations * grad_cube
        ).view(b, k, -1).sum(-1).view(b, k, 1, 1)

        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grad_sq / (denom + 1e-8)

        positive_grad = torch.relu(gradients)

        weights = (alpha * positive_grad).view(b, k, -1).sum(-1).view(b, k, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode='bilinear',
            align_corners=False
        )

        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        #cam = cam / (cam.mean() + 1e-8)
        #cam = torch.clamp(cam, 0, 1)

        return cam.detach(), logits

gradcampp = GradCAMpp(model, TARGET_LAYER)

def get_common_indices(classImg, beta, F, require_evstate=False, tau=50):
    """
    Returns sorted indices shared across all required files.
    """

    base_dir = os.path.join(
        os.getcwd(),
        "selected_fmaps",
        f"class{classImg}"
    )

    if beta == 0.99:
        pat = re.compile(
            rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.99_model{F}\.npz$"
        )
    else:
        betaList = list(str(beta))
        pat = re.compile(
            rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.{betaList[-1]}_model{F}\.npz$"
        )

    if not os.path.isdir(base_dir):
        print(f"Missing folder: {base_dir}")
        return []

    indices = []

    for fname in os.listdir(base_dir):

        m = pat.match(fname)

        if not m:
            continue

        idx = int(m.group(1))

        npz_path = os.path.join(
            base_dir,
            f"Img_class_{classImg}_idx_{idx}_beta_{beta}_model{F}.npz"
        )

        if not os.path.exists(npz_path):
            continue

        if require_evstate:

            evstate_path = os.path.join(
                os.getcwd(),
                "SavedStates",
                f"tau{tau}",
                f"class{classImg}",
                "states",
                f"evState_idx_{idx}_class{classImg}_beta_{beta}model{F}.npy"
            )

            if not os.path.exists(evstate_path):
                continue

        indices.append(idx)

    return sorted(indices)


def apply_cam(input_tensor, cam):
    if isinstance(cam, np.ndarray):
        cam = torch.from_numpy(cam)

    if cam.dim() == 3:
        cam = cam.unsqueeze(1)
    #binary_cam = (cam > 0.1).float()
    cam = torch.relu(cam)
    cam = torch.nn.functional.interpolate(
        cam,
        size=input_tensor.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    #binary_cam = (cam > 0.1).float()
    return input_tensor * cam

def calculate_percentage_drop(model, original_img, masked_img):

    with torch.no_grad():

        original_logit = model(original_img)
        original_probs = func.softmax(original_logit, dim=1)

        target_class = original_logit.argmax(dim=1).item()
        original_conf = original_probs[0, target_class].item()

        masked_logit = model(masked_img)
        masked_probs = func.softmax(masked_logit, dim=1)
        masked_conf = masked_probs[0, target_class].item()

        drop = max(
            0.0,
            (original_conf - masked_conf) /
            (original_conf + 1e-12)
        ) * 100.0

        return drop, original_conf, masked_conf, target_class


def grad_cam(features, gradients, subset_features=None):

    avg_gradients = gradients.mean(dim=(2, 3), keepdim=True)
    if subset_features is not None:

        gradcam = torch.relu(
            avg_gradients[:, subset_features] *
            features[:, subset_features]
        ).sum(dim=1, keepdim=True)

        alpha = avg_gradients.squeeze()[subset_features]

    else:
        gradcam = torch.relu(
            avg_gradients[:, :] *
            features[:, :]
        ).sum(dim=1, keepdim=True)

        alpha = avg_gradients

    gradcam = (
        gradcam - gradcam.min()
    ) / (
        gradcam.max() - gradcam.min() + 1e-8
    )

    return (
        gradcam.detach().cpu().numpy(),
        alpha.squeeze().detach().numpy()
    )


def evalQA(classImg, beta):
    tau = 50
    indices = get_common_indices(
        classImg,
        beta,
        F,
        require_evstate=True
    )
    if len(indices) == 0:
        print(f"No matching samples for class {classImg}")
        return None

    new_results = {
        "qa": {
            "drops": [],
            "total_qubits": [],
            "active_qubits": [],
            "wins": 0,
        }
    }
    n_processed = 0

    for n in tqdm(indices):
        npz_path = os.path.join(
            os.getcwd(),
            "selected_fmaps",
            f"class{classImg}",
            f"Img_class_{classImg}_idx_{n}_beta_{beta}_model{F}.npz"
        )

        evstate_path = os.path.join(
            os.getcwd(),
            "SavedStates",
            f"tau{tau}",
            f"class{classImg}",
            "states",
            f"evState_idx_{n}_class{classImg}_beta_{beta}model{F}.npy"
        )

        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"Failed loading {npz_path}: {e}")
            continue

        nonzero_indices = data["nonzero_indices"]
        cosine_similarity = data["coupling"]
        alpha = data["linear"]
        input_tensor_np = data["input_tensor"]

        n_qubits = int(nonzero_indices.shape[0])

        if n_qubits == 0:
            continue

        try:
            evState = np.load(evstate_path)
        except Exception as e:
            print(f"Failed loading state {evstate_path}: {e}")
            continue

        ham_zz = 0.0

        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                ham_zz += SpinOperator(
                    [('pz', i, 'pz', j)],
                    coupling=[cosine_similarity[i, j]],
                    size=n_qubits,
                    verbose=0
                ).qutip_op

        ham_ext_z = 0.0

        for i in range(n_qubits):
            ham_ext_z += SpinOperator(
                [('pz', i)],
                coupling=[-alpha[i]],
                size=n_qubits,
                verbose=0
            ).qutip_op

        ham_qubo = (1 - beta) * ham_zz + beta * ham_ext_z
        ham_qubo = qutip.Qobj(ham_qubo)

        P = (evState.conj() * evState).real

        shots = n_qubits ** 2

        selected_bitstring = sample_minimum_energy(
            P,
            shots,
            ham_qubo,
            n_qubits
        )

        current_selected_maps = nonzero_indices[
            np.nonzero(selected_bitstring)[0]
        ]

        extractor = FeatureExtractor()

        extractor.set_model(model)
        extractor.set_target_layer(model.reduce[-1])

        input_tensor = torch.from_numpy(input_tensor_np).float()

        if input_tensor.ndim == 3:

            if input_tensor.shape[-1] == 3:
                input_tensor = input_tensor.permute(2, 0, 1)

            input_tensor = input_tensor.unsqueeze(0)

        elif input_tensor.ndim == 4:

            if input_tensor.shape[-1] == 3:
                input_tensor = input_tensor.permute(0, 3, 1, 2)

        else:
            print(f"Skipping idx {n}: invalid shape {input_tensor.shape}")
            continue

        input_tensor = func.interpolate(
            input_tensor,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).to(DEVICE)

        assert input_tensor.shape == (1, 3, 224, 224)

        extractor.set_input_tensor(input_tensor, see_picture=False)

        extractor.register_hooks()

        features, gradients, pred = extractor.extract_features()

        extractor.remove_hooks()

        if np.max(current_selected_maps) >= features.shape[1]:
            print(
                f"Skipping idx {n}: invalid fmap index "
                f"(max={np.max(current_selected_maps)}, "
                f"features={features.shape[1]})"
            )
            continue

        try:

            cam_gc, _ = grad_cam(
                features=features.cpu(),
                gradients=gradients.cpu(),
                subset_features=current_selected_maps,
            )

        except Exception as e:
            print(f"Skipping idx {n}: grad_cam failed: {e}")
            continue

        heatmap = cam_gc[0, 0]
        heatmap_u8 = np.uint8(255 * heatmap)

        _, _, H, W = input_tensor.shape

        heatmap_img = Image.fromarray(
            heatmap_u8,
            mode="L"
        ).resize(
            (W, H),
            resample=Image.BILINEAR
        )

        heatmap = np.array(heatmap_img).astype(np.float32) / 255.0

        if heatmap.ndim == 3:
            heatmap = heatmap[..., 0]

        mask = torch.from_numpy(heatmap).float().to(input_tensor.device)
        mask = mask.unsqueeze(0).unsqueeze(0)
        #plt.imshow(mask.squeeze(0).permute(1,2,0))
        #plt.show()
        masked_gc = apply_cam(input_tensor, mask)
        plt.imshow(masked_gc.squeeze(0).permute(1, 2, 0))
        plt.show()

        try:
            drop_gc, _, _, _ = calculate_percentage_drop(
                model,
                input_tensor,
                masked_gc
            )
        except Exception as e:
            print(f"Skipping idx {n}: eval failed: {e}")
            continue

        if np.isfinite(drop_gc):

            new_results["qa"]["drops"].append(float(drop_gc))
            new_results["qa"]["total_qubits"].append(len(nonzero_indices))
            new_results["qa"]["active_qubits"].append(
                len(current_selected_maps)
            )

            new_results["qa"]["wins"] += 1
            n_processed += 1

    den = max(n_processed, 1)

    drops_arr = np.array(
        new_results["qa"]["drops"],
        dtype=np.float64
    )

    if len(drops_arr) > 0:
        new_results["qa"]["mean"] = float(np.mean(drops_arr))
        new_results["qa"]["std"] = float(np.std(drops_arr))
    else:
        new_results["qa"]["mean"] = np.nan
        new_results["qa"]["std"] = np.nan

    new_results["qa"]["win_pct"] = (
        new_results["qa"]["wins"] / den
    ) * 100.0

    print(
        f"QA: {new_results['qa']['mean']:.2f}% ± "
        f"{new_results['qa']['std']:.2f}%"
    )

    print(f"QA processed samples: {n_processed}")

    return new_results


def evalSA(classImg, beta):

    base_dir = os.path.join(os.getcwd(), "SA_fullresnet", f"class{classImg}")
    if beta == 0.99:
        pat = re.compile(rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.99_model{F}.npz$")
    else:
        betaList = list(str(beta))
        pat = re.compile(rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.{betaList[-1]}_model{F}.npz$")

    print(pat)

    if not os.path.isdir(base_dir):
        print(f"Missing folder: {base_dir}")
        return None

    indices = []
    for fname in os.listdir(base_dir):
        m = pat.match(fname)
        if m:
            indices.append(int(m.group(1)))

    indices = sorted(indices)

    if len(indices) == 0:
        print(f"No matching samples for class {classImg}")
        return None

    new_results = {"sa": {"drops": [],
                          "total_qubits": [],
                          "active_qubits": [],
                          }}
    processed = 0

    for idx in tqdm(indices, desc=f"Evaluating class {classImg}"):

        npz_path = os.path.join(
            base_dir,
            f"Img_class_{classImg}_idx_{idx}_beta_{beta}_model{F}.npz"
        )
        try:
            data = np.load(npz_path, allow_pickle=True)
            #print(data.keys())
        except FileNotFoundError:
            print(f"Skipping idx {idx}: file not found")
            continue

        #data = np.load(npz_path, allow_pickle=True)
        input_tensor_np = data["input_tensor"]
        input_tensor = torch.from_numpy(input_tensor_np).float()
        selected_feature_maps = data["selected_features"].astype(int)

        if "total_qubits" in data.files:
            total_qubits = int(data["total_qubits"])
        else:
            print(f"Missing total_qubits in {npz_path}")
            total_qubits = -1

        #total_qubits = data["total_qubits"]
        print("FMaps:", selected_feature_maps, "Num:", len(selected_feature_maps))
        print("Total qubits:", total_qubits)
        if len(selected_feature_maps) == 0:
            continue

        extractor = FeatureExtractor()
        extractor.set_model(model)

        extractor.set_target_layer(TARGET_LAYER)

        #features, gradients, pred = extractor.extract_features()

        if input_tensor.ndim == 3:
            # HWC -> CHW if needed
            if input_tensor.shape[-1] == 3:
                input_tensor = input_tensor.permute(2, 0, 1)
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 4:
            # BHWC -> BCHW if needed
            if input_tensor.shape[-1] == 3:
                input_tensor = input_tensor.permute(0, 3, 1, 2)
        else:
            print(f"Skipping idx {idx}: unexpected input_tensor shape {input_tensor.shape}")
            continue

        input_tensor = func.interpolate(
            input_tensor,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).to(DEVICE)

        #input_tensor.requires_grad_(True)
        #print(input_tensor.requires_grad_)
        extractor.set_input_tensor(input_tensor)
        #extractor.set_input_tensor(torch.from_numpy(input_tensor_np), see_picture=False)
        #extractor.register_hooks()
        #extractor.remove_hooks()
        extractor.register_hooks()

        #try:
        features, gradients, pred = extractor.extract_features()
        #except Exception as e:
        #    print(f"Skipping idx {n}: extractor failed: {e}")
        #    continue
        extractor.remove_hooks()
        try:
            cam_sa, _ = grad_cam(
                features=features.cpu(),
                gradients=gradients.cpu(),
                subset_features=selected_feature_maps,
            )
        except Exception as e:
            print(f"Skipping idx {idx}: grad_cam failed: {e}")
            continue
        masked_sa = apply_cam(input_tensor, cam_sa)

        """cam = torch.nn.functional.interpolate(
        torch.from_numpy(cam_gc),
        size=input_tensor.shape[-2:],
        mode='bilinear',
        align_corners=False
        )"""
        #print(input_tensor.shape)
        #print(cam.shape)

        #masked = input_tensor * cam
        #masked_input = masked

        try:
            drop, _, _, _ = calculate_percentage_drop(
                model, torch.from_numpy(input_tensor_np).unsqueeze(0), masked_sa
            )
        except Exception as e:
            print(f"Skipping idx {idx}: eval failed: {e}")
            continue

        if np.isfinite(drop):
            new_results["sa"]["drops"].append(float(drop))
            new_results["sa"]["total_qubits"].append(total_qubits)
            new_results["sa"]["active_qubits"].append(len(selected_feature_maps))

            processed += 1

    #den = max(processed, 1)
    drops = np.array(new_results["sa"]["drops"], dtype=np.float64)

    new_results["sa"]["mean"] = float(np.mean(drops)) if len(drops) else np.nan
    new_results["sa"]["std"] = float(np.std(drops)) if len(drops) else np.nan

    print(
        f"SA: {new_results['sa']['mean']:.2f}% ± {new_results['sa']['std']:.2f}% "
    )

    return new_results


def evaluate_sample(input_tensor):
    """
    Runs GradCAM and GradCAM++ on a single input.
    Returns drop dictionary.
    """
    drops = {}
    extractor = FeatureExtractor()
    extractor.set_model(model)

    extractor.set_target_layer(TARGET_LAYER)
    extractor.set_input_tensor(input_tensor)
    # extractor.set_input_tensor(torch.from_numpy(input_tensor_np), see_picture=False)
    # extractor.register_hooks()
    # extractor.remove_hooks()
    extractor.register_hooks()

    # try:
    features, gradients, pred = extractor.extract_features()
    # except Exception as e:
    #    print(f"Skipping idx {n}: extractor failed: {e}")
    #    continue
    extractor.remove_hooks()

    cam_gc, _ = grad_cam(
        features=features.cpu(),
        gradients=gradients.cpu(),
        subset_features=None,
    )
    #cam_gc, _ = gradcam(input_tensor)  ### :TODO HERE!
    cam_gcp, _ = gradcampp(input_tensor)

    masked_gc = apply_cam(input_tensor, cam_gc)
    masked_gcp = apply_cam(input_tensor, cam_gcp)
    #cam_gc, _ = gradcam(input_tensor)

    drop_gc, _, _, _ = calculate_percentage_drop(
        model, input_tensor, masked_gc
    )
    plt.imshow(masked_gc.squeeze(0).permute(1, 2, 0))
    plt.show()
    plt.imshow(masked_gcp.squeeze(0).permute(1, 2, 0))
    plt.show()
    drops['gradcam'] = float(drop_gc)

    drop_gcp, _, _, _ = calculate_percentage_drop(
        model, input_tensor, masked_gcp
    )
    drops['gradcampp'] = float(drop_gcp)

    return drops


def evaluate_class_CAMs(classImg, num_samples=None):
    if F == 16:
        base_dir = os.path.join(
            os.getcwd(),
            "selected_fmaps",
            f"class{classImg}"
        )

        if beta == 0.99:

            pat = re.compile(
                rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.99_model{F}\.npz$"
            )
        else:
            betaList = list(str(beta))
            pat = re.compile(
                rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.{betaList[-1]}_model{F}\.npz$"
            )
    elif F == 24:
        base_dir = os.path.join(
            os.getcwd(),
            "selected_fmaps",
            f"class{classImg}"
        )

        if beta == 0.99:

            pat = re.compile(
                rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.99_model{F}\.npz$"
            )
        else:
            betaList = list(str(beta))
            pat = re.compile(
                rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.{betaList[-1]}_model{F}\.npz$"
            )
    elif F == 512:
        base_dir = os.path.join(
            os.getcwd(),
            "SA_fullresnet",
            f"class{classImg}"
        )
        if beta == 0.99:
            pat = re.compile(
                rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.99_model{F}\.npz$"
            )
        else:
            betaList = list(str(beta))
            pat = re.compile(
                rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.{betaList[-1]}_model{F}\.npz$"
            )
    else:
        print(f"Unsupported F={F}")
        return None
    if not os.path.isdir(base_dir):
        print(f"Missing folder: {base_dir}")
        return None

    indices = get_common_indices(
        classImg,
        beta,
        F,
        require_evstate=False
    )
    """indices = []
    for fname in os.listdir(base_dir):
        m = pat.match(fname)
        if m:
            indices.append(int(m.group(1)))"""
    #indices = sorted(indices)
    if len(indices) == 0:
        print(f"No matching samples for class {classImg}")
        return None

    if num_samples is not None:
        indices = indices[:num_samples]

    print(
        f"\nClass {classImg} ({CLASS_NAMES[classImg]}) "
        f"- {len(indices)} samples"
    )
    results = {
        'gradcam': {'drops': []},
        'gradcampp': {'drops': []},
    }
    for n in tqdm(indices):
        npz_path = os.path.join(
            base_dir,
            f"Img_class_{classImg}_idx_{n}_beta_{beta}_model{F}.npz"
        )
        if not os.path.exists(npz_path):
            print(f"Missing file: {npz_path}")
            continue
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                input_np = data["input_tensor"]
        except Exception as e:
            print(f"Failed loading {npz_path}: {e}")
            continue
        input_tensor = torch.from_numpy(input_np).float()
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        if input_tensor.shape[1] != 3:
            input_tensor = input_tensor.permute(0, 3, 1, 2)
        input_tensor = torch.nn.functional.interpolate(
            input_tensor,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        ).to(DEVICE)
        try:
            drops = evaluate_sample(input_tensor)
        except Exception as e:
            print(f"Skipping {npz_path}: {e}")
            continue
        for method in drops:
            results[method]['drops'].append(
                float(drops[method])
            )
    for method in results:
        arr = np.array(
            results[method]['drops'],
            dtype=np.float64
        )
        results[method]['mean'] = (
            float(np.mean(arr))
            if len(arr) else np.nan
        )

        results[method]['std'] = (
            float(np.std(arr))
            if len(arr) else np.nan
        )
        results[method]['median'] = (
            float(np.median(arr))
            if len(arr) else np.nan
        )
    print("\nResults:")
    for method in results:
        r = results[method]
        print(
            f"{method:10s}: "
            f"{r['mean']:6.2f}% ± "
            f"{r['std']:6.2f}% | "
            f"median {r['median']:6.2f}%"
        )
    return results


"""all_results = {}
print("CAMs\n")
for c in range(NUM_CLASSES):
    all_results[c] = evaluate_class_CAMs(c)
    #print(all_results[c])"""

print("\nFINAL RESULTS\n")
"""
for class_idx, results in all_results.items():
    print(f"\nClass {class_idx} ({CLASS_NAMES[class_idx]})")
    for method in ['gradcam', 'gradcampp']:
        r = results[method]
        print(f"{method:10s} | "
            f"mean: {r['mean']:.2f}% | "
            f"std: {r['std']:.2f}% | "
            f"median: {r['median']:.2f}%"
        )"""

all_resultsQA = []
print("SAs\n")

for class_idx in range(NUM_CLASSES):
    results = evalSA(class_idx, beta)

    #results = evalQA(class_idx, beta)
    if results:
        all_resultsQA.append({
            'Class': CLASS_NAMES[class_idx],
            #'QA': results['qa'],
            'SA': results['sa'],
        })