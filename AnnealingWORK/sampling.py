from matplotlib import pyplot as plt
import numpy as np
import torch
from qutip_class import SpinOperator
import os
from src.feature_extraction_utils import FeatureExtractor, grad_cam
import torch.nn as nn
from torchvision import models


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

        self.classifier = nn.Linear(self.n_features * 14**2, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.reduce(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)



def basis(nqubits: int):
    num = 2 ** nqubits
    return np.array(
        [list(np.binary_repr(i, width= nqubits)) for i in range(num)],
        dtype=int
    )


def sample_minimum_energy(p_sampling, shots, Ht, qubits):
    instances = np.random.choice(np.arange(p_sampling.shape[0]), size=shots, p=p_sampling)
    energy_selected = Ht.diag().real[instances]
    selected_bistring_index = instances[np.argmin(energy_selected)]
    selected_bistring = basis(qubits)[selected_bistring_index]
    return selected_bistring


def bitstring_basis_msb(n_qubits: int) -> np.ndarray:
    num = 2**n_qubits
    arr = np.array([list(np.binary_repr(i, width=n_qubits)) for i in range(num)], dtype=int)
    return arr

def bitstring_basis_lsb(n_qubits: int) -> np.ndarray:
    msb = bitstring_basis_msb(n_qubits)
    return msb[:, ::-1]


def single_sampling(x):
    fmap_path = os.path.join(
        os.getcwd(),
        "selected_fmaps",
        f"class{imgClass}",
        f"Img_class_{imgClass}_idx_{x}_beta_{beta}_model{F}.npz"
    )

    expval_path = os.path.join(
        os.getcwd(),
        "SavedStates",
        f"tau{tau}",
        f"class{imgClass}",
        "et",
        f"expvalState_idx_{x}_class{imgClass}_beta_{beta}model{F}.npy"
    )

    evstate_path = os.path.join(
        os.getcwd(),
        "SavedStates",
        f"tau{tau}",
        f"class{imgClass}",
        "states",
        f"evState_idx_{x}_class{imgClass}_beta_{beta}model{F}.npy"
    )

    probs_path = os.path.join(
        os.getcwd(),
        "SavedStates",
        f"tau{tau}",
        f"class{imgClass}",
        "probs",
        f"probs_idx_{x}_class{imgClass}_beta_{beta}model{F}.npy"
    )

    fidelity_path = os.path.join(
        os.getcwd(),
        "SavedStates",
        f"tau{tau}",
        f"class{imgClass}",
        "fidelity",
        f"fidelity_idx_{x}_class{imgClass}_beta_{beta}model{F}.npy"
    )

    if not os.path.exists(fmap_path):
        print(f"Skipping x={x}: fmap file missing")

    if not os.path.exists(expval_path):
        print(f"Skipping x={x}: expval file missing")

    if not os.path.exists(evstate_path):
        print(f"Skipping x={x}: evState file missing")

    if not os.path.exists(probs_path):
        print(f"Skipping x={x}: probs file missing")

    if not os.path.exists(fidelity_path):
        print(f"Skipping x={x}: fidelity file missing")

    with np.load(fmap_path) as data:
        inputTensor = data["input_tensor"]
        nonzero_indices = data["nonzero_indices"]
        cosine_similarity = data["coupling"]
        alpha = data["linear"]

        print("Shape", cosine_similarity.shape)

        Psi = np.load(expval_path)
        evState = np.load(evstate_path)
        Probs = np.load(probs_path)
        Fidelity = np.load(fidelity_path)

        n_qubits = nonzero_indices.shape[0]
        # print("CS", cosine_similarity)
        ham_zz = 0.
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                ham_zz += SpinOperator([('pz', i, 'pz', j)], coupling=[cosine_similarity[i, j]], size=n_qubits,
                                       verbose=0).qutip_op

        ham_ext_z = 0.
        for i in range(n_qubits):
            ham_ext_z += SpinOperator([('pz', i)], coupling=[-alpha[i]], size=n_qubits, verbose=0).qutip_op

        ham_qubo = (1 - beta) * ham_zz + beta * ham_ext_z
        energies = ham_qubo.diag().real
        index = np.argsort(energies)

        prob = np.abs(evState.conj() * evState)**2
        prob = prob / np.sum(prob)


        # P /= P.sum()

        # P = np.abs(evState) ** 2
        # print("nonzero_indices.shape:", nonzero_indices.shape)
        # print("len(evState):", len(evState))
        # print("len(P):", len(P))
        # print("n_qubits from nonzero_indices:", nonzero_indices.shape[0])
        # print("n_qubits from evState:", int(np.log2(len(evState))))
        # P = P / P.sum()

        shots = n_qubits ** 2
        #### Emanuele's sampling ####
        samples = np.random.choice(np.arange(len(prob)), size=shots, p=prob)
        count = 0
        for sample in samples:
            if sample == index[0]:
                count +=1

        #selected_bitstring = sample_minimum_energy(P, shots, ham_qubo, n_qubits)
        bits_msb = bitstring_basis_msb(n_qubits)
        bitstring_features = nonzero_indices[np.nonzero(bits_msb[sample])[0]]
        print("Prob of success", count/shots)
        print("Bitstring:", bitstring_features)
        return (evState, bitstring_features, index)

        #bits_msb = bitstring_basis_msb(n_qubits)


def apply_cam(input_tensor, cam):
    if isinstance(cam, np.ndarray):
        cam = torch.from_numpy(cam)

    if cam.dim() == 3:
        cam = cam.unsqueeze(1)

    cam = torch.relu(cam)

    cam = torch.nn.functional.interpolate(
        cam,
        size=input_tensor.shape[-2:],
        mode='bilinear',
        align_corners=False
    )
    return input_tensor * cam

def evaluate_sample(input_tensor, bitstring):
    """
    Runs GradCAM and GradCAM++ on a single input.
    Returns drop dictionary.
    """
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
    #extractor.remove_hooks()
    cam_gc, _ = grad_cam(
        features=features.cpu(),
        gradients=gradients.cpu(),
        subset_features=bitstring,
    )
    # cam_gc, _ = gradcam(torch.from_numpy(inputT).unsqueeze(0))

    cam = torch.nn.functional.interpolate(
        torch.Tensor(cam_gc),
        # cam_gc,
        size=input_tensor.shape[-2:],
        mode='bilinear',
        align_corners=False
    )

    masked_gc = input_tensor * cam
    alpha = 0.4
    img = input_tensor.permute(1, 2, 0)
    cam2d = cam.squeeze()

    superimposed_gc = img + alpha * cam2d[..., None]
    superimposed_gc = superimposed_gc.numpy()
    import matplotlib.cm as cm
    heatmap_inferno_gc = cm.inferno(cam2d)[..., :3]
    superimposed_gc = (
            (1 - alpha) * input_tensor.permute(1, 2, 0) +
            alpha * heatmap_inferno_gc
    )
    superimposed_gc = np.clip(superimposed_gc, 0, 1)

    return superimposed_gc


if __name__ == "__main__":

    beta = 0.7; F = 24; tau = 50; imgClass = 3; idx = 7
    model = ReducedResNet18(n_features=F, num_classes=10)

    device = torch.device("cpu")
    # model = OriginalResNet18(num_classes=10, reduced_features=F)
    model.to(device)
    if F == 16:
        checkpoint = torch.load(
        os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep17_fullResnetMod16.pth"),
            map_location=device,
            weights_only=False)
    elif F == 24:
        checkpoint = torch.load(
        os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep20_fullResnetMod24.pth"),
            map_location=device,
            weights_only=False)
    elif F == 512:
        checkpoint = torch.load(
            os.path.join(os.getcwd(), "checkpoints", "checkpoint_ep20_fullResnetMod512.pth"),
            map_location=device,
            weights_only=False)
    else:
        raise ValueError("Not implemented")

    test_set = torch.load(os.path.join(os.getcwd(), "data", "data", "test_dataset.pt"),
                          map_location=device,
                          weights_only=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    TARGET_LAYER = model.reduce[-1]
    psi, bitstring, index = single_sampling(idx)

    plt.bar(np.arange(len(psi)), np.conj(psi)*psi)
    plt.axvline(index[0], color="red")
    plt.show()
    data = np.load(
        f"/Users/francescoaldoventurelli/qml/projet_one/AnnealingWORK/selected_fmaps/class{imgClass}/Img_class_{imgClass}_idx_{idx}_beta_{beta}_model{F}.npz",
        allow_pickle=True)
    inputT = torch.Tensor(data["input_tensor"])
    cam_overalyed = evaluate_sample(inputT, bitstring)
    plt.imshow(cam_overalyed)
    plt.show()



"""
    selected_maps = []
    for x in idx:
        fmap_path = os.path.join(
            os.getcwd(),
            "selected_fmaps",
            f"class{imgClass}",
            f"Img_class_{imgClass}_idx_{x}_beta_{beta}_model{F}.npz"
        )

        expval_path = os.path.join(
            os.getcwd(),
            "SavedStates",
            f"tau{tau}",
            f"class{imgClass}",
            "et",
            f"expvalState_idx_{x}_class{imgClass}_beta_{beta}model{F}.npy"
        )

        evstate_path = os.path.join(
            os.getcwd(),
            "SavedStates",
            f"tau{tau}",
            f"class{imgClass}",
            "states",
            f"evState_idx_{x}_class{imgClass}_beta_{beta}model{F}.npy"
        )

        probs_path = os.path.join(
            os.getcwd(),
            "SavedStates",
            f"tau{tau}",
            f"class{imgClass}",
            "probs",
            f"probs_idx_{x}_class{imgClass}_beta_{beta}model{F}.npy"
        )

        fidelity_path = os.path.join(
            os.getcwd(),
            "SavedStates",
            f"tau{tau}",
            f"class{imgClass}",
            "fidelity",
            f"fidelity_idx_{x}_class{imgClass}_beta_{beta}model{F}.npy"
        )

        if not os.path.exists(fmap_path):
            print(f"Skipping x={x}: fmap file missing")
            continue

        if not os.path.exists(expval_path):
            print(f"Skipping x={x}: expval file missing")
            continue

        if not os.path.exists(evstate_path):
            print(f"Skipping x={x}: evState file missing")
            continue

        if not os.path.exists(probs_path):
            print(f"Skipping x={x}: probs file missing")
            continue

        if not os.path.exists(fidelity_path):
            print(f"Skipping x={x}: fidelity file missing")
            continue

        with np.load(fmap_path) as data:
            inputTensor = data["input_tensor"]
            nonzero_indices = data["nonzero_indices"]
            cosine_similarity = data["coupling"]
            alpha = data["linear"]

            print("Shape", cosine_similarity.shape)

        Psi = np.load(expval_path)
        evState = np.load(evstate_path)
        Probs = np.load(probs_path)
        Fidelity = np.load(fidelity_path)

        n_qubits = nonzero_indices.shape[0]
        #print("CS", cosine_similarity)
        ham_zz=0.
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                ham_zz+=SpinOperator([('pz',i,'pz',j)],coupling=[cosine_similarity[i,j]],size=n_qubits,verbose=0).qutip_op

        ham_ext_z=0.
        for i in range(n_qubits):
            ham_ext_z+=SpinOperator([('pz',i)],coupling=[-alpha[i]],size=n_qubits,verbose=0).qutip_op

        ham_qubo=(1-beta)*ham_zz+beta*ham_ext_z
        energies = ham_qubo.diag().real
        index = np.argsort(energies)

        P = evState.conj() * evState
        P = P.real

        #P /= P.sum()

        #P = np.abs(evState) ** 2
        #print("nonzero_indices.shape:", nonzero_indices.shape)
        #print("len(evState):", len(evState))
        #print("len(P):", len(P))
        #print("n_qubits from nonzero_indices:", nonzero_indices.shape[0])
        #print("n_qubits from evState:", int(np.log2(len(evState))))
        #P = P / P.sum()

        shots = n_qubits**2
        #### Emanuele's sampling ####


        selected_bitstring = sample_minimum_energy(P, shots, ham_qubo, n_qubits)

        print("Bitstring", selected_bitstring)
        print("Features activated", nonzero_indices[np.nonzero(selected_bitstring)[0]])

        bits_msb = bitstring_basis_msb(n_qubits)
        #print("Selected maps", nonzero_indices[np.nonzero(bits_msb[selected_bitstring])[0]])
        selected_maps.append(nonzero_indices[np.nonzero(selected_bitstring)[0]])

        #selected_maps.append(nonzero_indices[np.nonzero(bits_msb[selected_bitstring])[0]])"""


    #np.save(os.path.join(os.getcwd(), "selected_fmaps", f"class{imgClass}", f"histClass{imgClass}-mostFrequent_beta{beta}_F{F}.npy"), np.array(selected_maps, dtype=object))