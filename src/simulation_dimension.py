import os
import re
import pickle
from qutip_class import *
import qutip
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, expm_multiply
from tqdm import tqdm
from pathlib import Path
import sys


def qutip_to_scipy_sparse(qobj):
    """Convert QuTiP Qobj to scipy.sparse.csr_matrix."""
    data = qobj
    return csr_matrix((data.data, data.indices, data.indptr), shape=data.shape)


def qubo_x(i, n):
    """Return the QUBO variable x_i = |1><1| for qubit i in n-qubit space."""
    ops = [qutip.qeye(2)] * n
    ops[i] = (qutip.qeye(2) - qutip.sigmaz()) / 2
    return qutip.tensor(ops)


def build_qubo_hamiltonian(alpha, cosine_similarity, beta):
    n = len(alpha)
    H = 0 * qubo_x(0, n)

    for i in range(n):
        H += -beta * alpha[i] * qubo_x(i, n)

    for i in range(n):
        for j in range(i+1, n):
            H += (1 - beta) * cosine_similarity[i, j] * qubo_x(i, n) * qubo_x(j, n)
    return H


def index_to_bits(idx, n_qubits):
    return np.array(list(np.binary_repr(idx, width=n_qubits)), dtype=int)


#### commenti di Ema
##
#

classImg = int(sys.argv[2])
# indexImg = list(range(20))
beta = 0.7
tau = int(sys.argv[1])
dt = 0.01
n_levels = 2
F = 24
path = os.getcwd()
timesteps = int(tau / dt)
time = np.linspace(0, tau, timesteps)


def bitstring_basis_msb(n_qubits: int) -> np.ndarray:
    """Return (2**n_qubits, n_qubits) array of 0/1 bitstrings.
       MSB-first: qubit 0 is the leftmost column — matches QuTiP tensor(basis(...)) ordering."""
    num = 2 ** n_qubits
    # each row is the binary representation (MSB first)
    arr = np.array([list(np.binary_repr(i, width=n_qubits)) for i in range(num)], dtype=int)
    return arr


def annealing_sim(ham_transverse, ham_qubo, psi0) -> np.ndarray:
    fidelity = np.zeros(timesteps)
    T, Q = ham_transverse.data_as("csr_matrix"), ham_qubo.data_as("csr_matrix")
    T, Q = T.astype(np.complex128, copy=False), Q.astype(np.complex128, copy=False)
    s_arr = time / tau
    psi = psi0[:, 0].astype(np.complex128, copy=False)

    for r, s in tqdm(enumerate(s_arr), total=len(s_arr)):

        H_t = (1 - s) * T + s * Q
        psi = expm_multiply(-1j * H_t * dt, psi)

        energies_t, psi_levels = eigsh(H_t, k=n_levels, which='SA')
        ascendinOrder = np.argsort(energies_t)
        psi_levels = psi_levels[:, ascendinOrder]

        #probs[r, :] = np.abs(psi.conj().T @ psi_levels) ** 2

        fidelity[r] = np.abs(psi.conj().T @ psi_levels[:, 0]) ** 2
    return fidelity


def ensure_output_dirs(base_path: str, tau: int, class_id: int, F: int) -> dict:
    #root = Path(base_path) / "quantumStateEvolution" / f"tau{tau}" / f"class{class_id}" / f"model{F}"
    root = Path(base_path) / "Dictonaries" / f"tau{tau}" / f"class{class_id}"
    dirs = {
        "root": root,
        "fidelity": root / "fidelity"
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


folder = os.path.join(path, "selected_fmaps", f"class{classImg}")
pat = re.compile(rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.7_model{F}.npz$")

existing_indices = []
for fname in os.listdir(folder):
    m = pat.match(fname)
    if m:
        existing_indices.append(int(m.group(1)))


existing_indices = sorted(set(existing_indices))
print(existing_indices)
dirs = ensure_output_dirs(path, tau, classImg, F)

probs_dict = {
    "img_idx": [],
    "dim": [],
    "fidelity": []
    }


for n in existing_indices:
    data = np.load((os.path.join(path, "selected_fmaps", f"class{classImg}",
                                 f"Img_class_{classImg}_idx_{n}_beta_0.7_model{F}.npz")),
        allow_pickle=True)
    print(n)
    print(data)

    nonzero_indices = data['nonzero_indices']
    cosine_similarity = data['coupling']
    alpha = data['linear']
    input_tensor = data['input_tensor']
    print(alpha)
    #print(cosine_similarity.shape)
    n_qubits = nonzero_indices.shape[0]
    H_qubo = build_qubo_hamiltonian(alpha, cosine_similarity, beta)
    bits_msb = bitstring_basis_msb(n_qubits)
    ham_zz = 0.
    for i in range(n_qubits):
        for j in range(n_qubits):
            ham_zz += SpinOperator([('pz', i, 'pz', j)], coupling=[cosine_similarity[i, j]], size=n_qubits,
                                   verbose=0).qutip_op

    ham_ext_z = 0.
    for i in range(n_qubits):
        ham_ext_z += SpinOperator([('pz', i)], coupling=[-alpha[i]], size=n_qubits, verbose=0).qutip_op

    ham_qubo = (1 - beta) * ham_zz + beta * ham_ext_z
    ham_qubo = qutip.Qobj(ham_qubo)
    energies = ham_qubo.diag().real
    ham_transverse = SpinOperator([("x", i) for i in range(n_qubits)], coupling=[1/n_qubits]*n_qubits, size=n_qubits, verbose=0).qutip_op
    idx = np.argsort(energies)

    e, psi0 = eigsh(ham_transverse.data_as('csr_matrix').astype(np.complex128, copy=False), k=1, which='SA')
    print(e)
    #print(n_qubits)
    psi_gs=np.zeros(2**n_qubits)
    psi_gs[idx[0]]=1.
    print(f"Img: {n} - Nqubits: {n_qubits}")
    probs = annealing_sim(ham_transverse, ham_qubo, psi0)

    probs_dict["img_idx"].append(n)
    probs_dict["dim"].append(n_qubits)
    probs_dict["fidelity"].append(probs)

    #dataframe = pd.DataFrame.from_dict(probs_dict).set_index("img_idx")
    #dataframe.to_csv(dirs["probs"] / f"probs_Dict_class{classImg}_beta_{beta}model{F}.csv")

    #np.savez(dirs["probs"] / f"fidelity_Dict_class{classImg}_beta_{beta}model{F}.npz", probs_dict, allow_pickle=True)
    with open(os.path.join(dirs["fidelity"], f"fidelity_Dict_class{classImg}_beta_{beta}model{F}.pkl"), "wb") as f:
        pickle.dump(probs_dict, f)
