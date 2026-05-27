from matplotlib import pyplot as plt
import numpy as np
#import torch
from qutip_class import SpinOperator
import os


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


if __name__ == "__main__":

    beta = 0.7; F = 32; tau = 50; imgClass = 5; idx = list(range(20))


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

        #selected_maps.append(nonzero_indices[np.nonzero(bits_msb[selected_bitstring])[0]])

    """flat = np.concatenate(selected_maps, axis=0)
    hist, bins = np.histogram(flat, range=(0, F), bins=F)
    hist = hist / hist.sum()
    plt.bar(range(F), hist)
    plt.xticks(range(F))
    plt.xlabel("Value")
    plt.ylabel("Normalized frequency")
    #plt.legend(title=f"Class = {imgClass}")
    plt.show()
    np.save(os.path.join(os.getcwd(), "histos", f"histClass{imgClass}-mostFrequent_beta{beta}_F{F}.npy"), hist)
    """

    np.save(os.path.join(os.getcwd(), "selected_fmaps", f"class{imgClass}", f"histClass{imgClass}-mostFrequent_beta{beta}_F{F}.npy"), np.array(selected_maps, dtype=object))