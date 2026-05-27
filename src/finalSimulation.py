import os
import re
from qutip_class import *
import qutip
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm
from pathlib import Path
import sys


def qutip_to_scipy_sparse(qobj):
    """Convert QuTiP Qobj to scipy.sparse.csr_matrix."""
    data = qobj
    return csr_matrix((data.data, data.indices, data.indptr), shape=data.shape)


def qubo_x(i: int, n: int) -> qutip.Qobj:
    """Return the QUBO variable x_i = |1><1| for qubit i in n-qubit space."""
    ops = [qutip.qeye(2)] * n
    ops[i] = (qutip.qeye(2) - qutip.sigmaz()) / 2
    return qutip.tensor(ops)


def build_qubo_hamiltonian(alpha: dict, cosine_similarity: dict, beta: float):
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
#classImg = 5
# indexImg = list(range(20))
beta = 0.7
#tau = 30
tau = int(sys.argv[1])
dt = 0.01
n_levels = 2
F = 32
path = os.getcwd()

timesteps = int(tau / dt)
time = np.linspace(0, tau, timesteps)

print("Features:", F)

def bitstring_basis_msb(n_qubits: int) -> np.ndarray:
    """Return (2**n_qubits, n_qubits) array of 0/1 bitstrings.
       MSB-first: qubit 0 is the leftmost column — matches QuTiP tensor(basis(...)) ordering."""
    num = 2 ** n_qubits
    # each row is the binary representation (MSB first)
    arr = np.array([list(np.binary_repr(i, width=n_qubits)) for i in range(num)], dtype=int)
    return arr


def annealing_sim(ham_transverse, ham_qubo, psi0):
    spectrum = np.zeros((timesteps, n_levels), dtype=np.float64)
    probs = np.zeros((timesteps, n_levels), dtype=np.float64)
    e_t = np.zeros((timesteps), dtype=np.float64)
    fidelity = np.zeros((timesteps))
    # psis = []
    T, Q = ham_transverse.data_as("csr_matrix"), ham_qubo.data_as("csr_matrix")
    T, Q = T.astype(np.complex128, copy=False), Q.astype(np.complex128, copy=False)
    s_arr = time / tau
    psi = psi0[:, 0].astype(np.complex128, copy=False)
    #v0 = None
    for r, s in tqdm(enumerate(s_arr), total=len(s_arr)):

        H_t = (1 - s) * T + s * Q
        psi = expm_multiply(-1j * H_t * dt, psi)

        energies_t, psi_levels = eigsh(H_t, k=n_levels, which='SA')
        #energies_t, psi_levels = eigsh(H_t, k=n_levels, which='SA', v0=v0)
        ascendinOrder = np.argsort(energies_t)
        energies_t = energies_t[ascendinOrder].real
        psi_levels = psi_levels[:, ascendinOrder]
        e_t[r] = np.vdot(psi, H_t @ psi).real
        spectrum[r, :] = energies_t
        probs[r, :] = np.abs(psi.conj().T @ psi_levels) ** 2
        fidelity[r] = np.abs(psi.conj().T @ psi_gs) ** 2
        #v0 = psi_levels[:, 0]

    psi_fin = psi
    return spectrum, probs, psi_fin, e_t, fidelity


def ensure_output_dirs(base_path: str, tau: int, class_id: int, F: int) -> dict:
    #root = Path(base_path) / "quantumStateEvolution" / f"tau{tau}" / f"class{class_id}" / f"model{F}"
    root = Path(base_path) / "SavedStates" / f"tau{tau}" / f"class{class_id}"
    dirs = {
        "root": root,
        "spectrums": root / "spectrums",
        "probs": root / "probs",
        "et": root / "et",
        "states": root / "states",
        "fidelity": root / "fidelity",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


# Energies, State, Et = [], [], []
# folder = os.path.join(path, f"newFeatureExtracted/class{classImg}model{F}")

folder = os.path.join(path, "selected_fmaps", f"class{classImg}")
betaList = list(str(beta))
pat = re.compile(rf"^Img_class_{classImg}_idx_(\d+)_beta_0\.{betaList[-1]}_model{F}.npz$")
print(pat)
existing_indices = []
for fname in os.listdir(folder):
    m = pat.match(fname)
    #print(m)
    if m:
        existing_indices.append(int(m.group(1)))


existing_indices = sorted(set(existing_indices))
print(existing_indices)
dirs = ensure_output_dirs(path, tau, classImg, F)

for n in existing_indices:
    data = np.load((os.path.join(path, "selected_fmaps", f"class{classImg}",
                                 f"Img_class_{classImg}_idx_{n}_beta_{beta}_model{F}.npz")),
        allow_pickle=True)

    nonzero_indices = data['nonzero_indices']
    cosine_similarity = data['coupling']
    alpha = data['linear']
    input_tensor = data['input_tensor']
    n_qubits = nonzero_indices.shape[0]
    H_qubo = build_qubo_hamiltonian(alpha, cosine_similarity, beta)
    print("Alphas", alpha)
    bits_msb = bitstring_basis_msb(n_qubits)
    ham_zz = 0.
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
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
    energies, probs, evState, e_t, fidelity = annealing_sim(ham_transverse, ham_qubo, psi0)

    np.save(dirs["spectrums"] / f"energies_idx_{n}_class{classImg}_beta_{beta}model{F}.npy", energies)
    np.save(dirs["probs"] / f"probs_idx_{n}_class{classImg}_beta_{beta}model{F}.npy", probs)
    np.save(dirs["et"] / f"expvalState_idx_{n}_class{classImg}_beta_{beta}model{F}.npy", e_t)
    np.save(dirs["states"] / f"evState_idx_{n}_class{classImg}_beta_{beta}model{F}.npy", evState)
    np.save(dirs["fidelity"] / f"fidelity_idx_{n}_class{classImg}_beta_{beta}model{F}.npy", fidelity)