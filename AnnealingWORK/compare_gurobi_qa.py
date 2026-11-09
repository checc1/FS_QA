import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from sampling import ReducedResNet18, single_sampling, evaluate_sample
from collections import Counter


gurobi_folder = os.path.join(os.getcwd(), 'gurobi_results')
qa_folder = os.path.join(os.getcwd(), 'selected_fmaps')
sa_folder = os.path.join(os.getcwd(), 'sa_results')

def results_pre_class(class_index: int):
    tau = 50;n_index = 19;beta = 0.7;F = 24
    gurobi_sol, annealing_sol, sa_sol = [], [], []
    device = torch.device("cpu")

    model = ReducedResNet18(n_features=F, num_classes=10).to(device)

    checkpoint_paths = {
        16: "checkpoint_ep17_fullResnetMod16.pth",
        24: "checkpoint_ep20_fullResnetMod24.pth",
        32: "checkpoint_ep6_fullResnetMod32.pth",
        512: "checkpoint_ep20_fullResnetMod512.pth"
    }

    if F not in checkpoint_paths:
        raise ValueError("Not implemented")

    checkpoint = torch.load(
        os.path.join(os.getcwd(), "checkpoints", checkpoint_paths[F]),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    for i in range(n_index):
        bitstring_gurobi = np.load(
            os.path.join(gurobi_folder, f'F{F}', f'optimal_solutions_{class_index}_idx_{i}_beta_{beta}.npz')
        )
        bitstring_sa = np.load(
            os.path.join(sa_folder, f'F{F}', f'optimal_solutions_{class_index}_idx_{i}_beta_{beta}.npz')
        )
        psi, bitstring, index = single_sampling(n_index, class_index, tau, beta, F)

        gurobi_sol.append(tuple(np.array(bitstring_gurobi['solutions']).ravel()))
        annealing_sol.append(tuple(np.array(bitstring).ravel()))
        sa_sol.append(tuple(np.array(bitstring_sa['solutions']).ravel()))

    gurobi_sol = [tuple(int(val) for val in x) for x in gurobi_sol]
    annealing_sol = [tuple(int(val) for val in x) for x in annealing_sol]
    sa_sol = [tuple(int(val) for val in x) for x in sa_sol]

    gurobi_freq = Counter(gurobi_sol)
    annealing_freq = Counter(annealing_sol)
    sa_freq = Counter(sa_sol)

    gurobi_freq_str = {str(k): v for k, v in gurobi_freq.items()}
    annealing_freq_str = {str(k): v for k, v in annealing_freq.items()}
    sa_freq_str = {str(k): v for k, v in sa_freq.items()}

    all_solutions = sorted(list(set(gurobi_freq_str.keys()) | set(annealing_freq_str.keys())
                                | set(sa_freq_str.keys())))
    all_solutions_xticks = [''.join(map(str, sol)) for sol in all_solutions]
    gurobi_counts = [gurobi_freq_str.get(sol, 0) for sol in all_solutions]
    annealing_counts = [annealing_freq_str.get(sol, 0) for sol in all_solutions]
    sa_counts = [sa_freq_str.get(sol, 0) for sol in all_solutions]

    x = np.arange(len(all_solutions))
    width = 0.3

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.bar(x - width, gurobi_counts, width, label='Gurobi', color='skyblue', edgecolor='w')
    ax.bar(x, annealing_counts, width, label='QA', color='orange', edgecolor='w')
    ax.bar(x + width, sa_counts, width, label='SA', color='red', edgecolor='w')

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Bitstring')
    ax.set_xticks(x)
    ax.set_xticklabels(all_solutions_xticks, rotation=45, ha='right')
    ax.legend(edgecolor="k", frameon=True, title=f'Class {class_index}')

    plt.tight_layout()
    plt.show()


def result_on_image(class_index: int, n_index: int, method: str = "gurobi"):
    F = 24;device = torch.device("cpu");beta = 0.7;tau = 50
    model = ReducedResNet18(n_features=F, num_classes=10).to(device)

    checkpoint_paths = {
        16: "checkpoint_ep17_fullResnetMod16.pth",
        24: "checkpoint_ep20_fullResnetMod24.pth",
        32: "checkpoint_ep6_fullResnetMod32.pth",
        512: "checkpoint_ep20_fullResnetMod512.pth"
    }

    if F not in checkpoint_paths:
        raise ValueError("Not implemented")

    checkpoint = torch.load(
        os.path.join(os.getcwd(), "checkpoints", checkpoint_paths[F]),
        map_location=device,
        weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    TARGET_LAYER = model.reduce[-1]


    if method == "gurobi":
        data = np.load(
            os.path.join(gurobi_folder, f'F{F}', f'optimal_solutions_{class_index}_idx_{n_index}_beta_{beta}.npz')
        )
        res = evaluate_sample(input_tensor=torch.from_numpy(data['input_tensor']), bitstring=data['solutions'],
                              model=model, TARGET_LAYER=TARGET_LAYER)
    elif method == "sa":
        data = np.load(
            os.path.join(sa_folder, f'F{F}', f'optimal_solutions_{class_index}_idx_{n_index}_beta_{beta}.npz')
        )
        res = evaluate_sample(input_tensor=torch.from_numpy(data['input_tensor']), bitstring=data['solutions'],
                              model=model, TARGET_LAYER=TARGET_LAYER)
    elif method == "qa":
        input_tensor = np.load(os.path.join(qa_folder, f'class{class_index}',
                                    f'Img_class_{class_index}_idx_{n_index}_beta_{beta}_model{F}.npz'))['input_tensor']
        _, bitstring, _ = single_sampling(n_index, class_index, tau, beta, F)
        print(input_tensor.shape)
        res = evaluate_sample(input_tensor=torch.from_numpy(input_tensor), bitstring=bitstring,
                              model=model, TARGET_LAYER=TARGET_LAYER)
    else:
        raise ValueError(f"Method '{method}' is not implemented yet.")

    return res



"""if __name__ == "__main__":
    class_index = 5

    res = result_on_image(class_index, n_index=18, method="gurobi")
    plt.imshow(res)
    plt.show()
    plt.bar(np.arange(len(psi)), np.real(np.conj(psi)*psi))
    plt.axvline(index[0], color="red", alpha=0.6)
    plt.show()
    data = np.load(os.path.join(os.getcwd(),
                                "selected_fmaps",
                                f"class{imgClass}/Img_class_{imgClass}_idx_{idx}_beta_{beta}_model{F}.npz"),
        allow_pickle=True)

    inputT = torch.Tensor(data["input_tensor"])
    cam_overalyed = evaluate_sample(inputT, bitstring, model, TARGET_LAYER)
    plt.imshow(cam_overalyed)
    plt.show()"""