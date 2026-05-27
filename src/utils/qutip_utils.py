from typing import List, Tuple,Dict,Optional
import numpy as np
import qutip
from scipy.sparse.linalg import eigsh
from qutip import basis, tensor
from tqdm import tqdm
from scipy.sparse.linalg import expm_multiply


def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


class ManyBodyQutipOperator:
    def __init__(
        self,
        local_op: Optional[List[Tuple[qutip.Qobj]]] = None,
        description: Optional[str] = None,
        verbose: int = 0,
    ) -> None:
        """_summary_

        Args:
            local_op (Optional[List[qutip.Qobj]], optional): _description_. Defaults to None.
            description (Optional[str], optional): _description_. Defaults to None.
        """

        self.__get_qutip_op(local_op)

        self.description = description
        self.verbose = verbose

    @property
    def qutip_op(self):
        return self._qutip_op

    @property
    def description(self):
        return self._description

    @description.setter
    def description(self, comment: str):
        self._description = comment

    @qutip_op.setter
    def qutip_op(self, mbop: qutip.Qobj):
        self._qutip_op: qutip.Qobj = mbop

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value: int):
        if not (value in [0, 1]):
            raise ValueError(
                "verbose should be either 0 (just string description) or 1 (matrix printout) \n"
            )
        self._verbose = value

    def __get_qutip_op(self, local_op: List[List[qutip.Qobj]]):

        if local_op is not (None):
            for i, ops in enumerate(local_op):
                for j, op in enumerate(ops):
                    if type(op) != qutip.qobj.Qobj:
                        raise TypeError(
                            f"Element {i} is not a Qutip Object Qobj ({type(op)} instead)"
                        )
                    if j == 0:
                        mbop = op
                    else:
                        mbop = qutip.tensor(mbop, op)
                if i == 0:
                    self.qutip_op = mbop
                else:
                    self.qutip_op = self.qutip_op + mbop

    def expect_value(self, psi: qutip.Qobj) -> float:
        return qutip.expect(self.qutip_op, psi)

    def __str__(self) -> str:
        if self.verbose == 0:
            return f"{self.description} \n"
        else:
            return f"{self.description} \n {self.qutip_op} \n"


class SpinOperator(ManyBodyQutipOperator):
    def __init__(
        self,
        index: List[Tuple],
        coupling: List,
        size: int,
        verbose: int = 0,
    ) -> None:

        super().__init__()

        # dictionary for the conversion string to local operator
        self._local_obs_dict = {
            "x": qutip.sigmax(),
            "y": qutip.sigmay(),
            "z": qutip.sigmaz(),
            "+": qutip.sigmap(),
            "-": qutip.sigmam(),
            "id": qutip.identity(2),
            "pz": qutip.qeye(2) - qutip.sigmaz(),
            "px": qutip.qeye(2) + qutip.sigmax(),
            "py": qutip.qeye(2) + qutip.sigmay(),
        }

        # size of the system
        self.size = size
        # list of tuples (indices, local operator)
        self.index = index
        # coupling constants
        self.coupling = coupling

        # getter of the qutip op
        # in this subclass
        self.__get_qutip_op()

        self.verbose = verbose

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: List[Tuple]):
        for k, tuple_indices in enumerate(index):
            for direction, idx in pairwise(tuple_indices):
                # check the direction
                if not (direction in self._local_obs_dict.keys()):
                    raise ValueError(
                        f"local operator string not defined -> {direction} index -> {idx}"
                    )

                if idx > self.size - 1:
                    raise ValueError(f"error, index larger than the number of sites")

        self._index = index

    @property
    def coupling(self):
        return self._coupling

    @coupling.setter
    def coupling(self, coupling: List):
        self._coupling = coupling

    @property
    def qutip_op(self):
        return self._qutip_op

    @qutip_op.setter
    def qutip_op(self, mbop: qutip.Qobj):

        if mbop.data.shape != (2 ** self.size, 2 ** self.size):
            raise ValueError(
                f"size mismatch -> l={self.size} effective l={mbop.data.shape}"
            )
        if mbop.dims != (
            [[2 for i in range(self.size)], [2 for i in range(self.size)]]
        ):
            raise ValueError(
                f"dimension mismatch -> not a qubit representation ({mbop.dims})"
            )

        self._qutip_op = mbop

    def __get_qutip_op(
        self,
    ):
        self._description: str = "(couplings, operator) -> \n"
        for k, tuple_indices in enumerate(self.index):

            # initialize the description
            self._description = (
                self._description + f" ( {self.coupling[k]} ,  {tuple_indices} ) \n"
            )

            coupling = self.coupling[k]
            # initialize the SpinOperator
            op_dict: Dict = {}
            indices: List = []
            for direction, idx in pairwise(tuple_indices):

                # define the given
                # local label operator

                if idx in op_dict.keys():
                    op_dict[idx] = op_dict[idx] * self._local_obs_dict[direction]
                else:
                    op_dict[idx] = self._local_obs_dict[direction]
                    indices.append(idx)

                # we fix the first part of the chain with
                # an identity operator
            if not (0 in op_dict.keys()):
                indices.append(0)
                op_dict[0] = self._local_obs_dict["id"]

                # and the last part with another identity
                # operator
            if not (self.size - 1 in op_dict.keys()):
                indices.append(self.size - 1)
                op_dict[self.size - 1] = self._local_obs_dict["id"]

            # order the indices
            indices.sort()
            # method for qutip.tensor optimization by Ewen Lawrence https://github.com/ewenlawrence
            # dumb index
            jdx: int = 0

            for i, idx in enumerate(indices):
                # if the sites are not nearest neighbours
                # create a identity with dim 2**(i-j)

                if idx - jdx > 1:
                    identity = qutip.identity(2 ** (idx - jdx - 1))
                    chain_oper = qutip.tensor(identity, op_dict[idx])
                # otherwise define the operator
                else:
                    chain_oper = op_dict[idx]
                # if k == 1:
                #     print("partial chain oper=", chain_oper, k)

                if i == 0:
                    many_body_op = chain_oper
                else:
                    many_body_op = qutip.tensor(many_body_op, chain_oper)

                jdx = idx

            # reshape the dimension of the qutip
            # object
            many_body_op = qutip.Qobj(
                many_body_op.data,
                dims=[[2 for i in range(self.size)], [2 for i in range(self.size)]],
            )

            # sum each direction
            if k == 0:
                self.qutip_op = many_body_op * coupling
            else:
                self.qutip_op = self.qutip_op + many_body_op * coupling

class QuboAnnealingOnQutip:
    
    def __init__(self,beta:float=0.5):
        
        self._nqubits=None
        self._target_hamiltonian=None
        self._driver_hamiltonian=None
        
        self._beta=beta
        
        pass


        

    def get_target_hamiltonian_from_data(self,data: Dict):
        
        self._nonzero_indices=data['nonzero_indices']
        self._nqubits=self._nonzero_indices.shape[0]
        
        
        self._coupling=data['coupling']
        self._linear=data['linear']
        self._input_tensor=data['input_tensor']
        
        
        self.basis=self.__get_basis_states()
        
        
        ham_zz=0.
        for i in range(self.nqubits):
            for j in range(self.nqubits):
                ham_zz+=SpinOperator([('qz',i,'qz',j)],coupling=[self._coupling[i,j]],size=self.nqubits,verbose=1).qutip_op
        
        ham_ext_z=0.
        for i in range(self.nqubits):
            ham_ext_z+=SpinOperator([('qz',i)],coupling=[-self._linear[i]],size=self.nqubits,verbose=0).qutip_op
        self._target_hamiltonian=(1-self.beta)*ham_zz+self._beta*ham_ext_z
        
    @property
    def beta(self):
        return self._beta
    
    @beta.setter
    def beta(self,value:float):
        if not (0.0<=value<=1.0):
            raise ValueError("Beta must be in [0,1]")
        self._beta=value
        if self._target_hamiltonian is not None:
            self.get_target_hamiltonian_from_data({'nonzero_indices':self._nonzero_indices,
                                                   'coupling':self._coupling,
                                                   'linear':self._linear,
                                                   'input_tensor':self._input_tensor})
    
    @property
    def nqubits(self):
        return self._nqubits
    @nqubits.setter
    def nqubits(self,value:int):
        if value<=0:
            raise ValueError("Number of qubits must be positive.")
        self._nqubits=value
        self.basis=self.__get_basis_states()
        
    @property
    def coupling(self):
        return self._coupling
    @coupling.setter
    def coupling(self,value:np.ndarray):
        if value.shape[0]!=self.nqubits or value.shape[1]!=self.nqubits:
            raise ValueError("Coupling matrix must be of shape (nqubits, nqubits).")
        self._coupling=value
        if self._target_hamiltonian is not None:
            self.get_target_hamiltonian_from_data({'nonzero_indices':self._nonzero_indices,
                                                   'coupling':self._coupling,
                                                   'linear':self._linear,
                                                   'input_tensor':self._input_tensor})
    @property
    def linear(self):
        return self._linear
    @linear.setter
    def linear(self,value:np.ndarray):
        if value.shape[0]!=self.nqubits:
            raise ValueError("Linear array must be of shape (nqubits,).")
        self._linear=value
        if self._target_hamiltonian is not None:
            self.get_target_hamiltonian_from_data({'nonzero_indices':self._nonzero_indices,
                                                   'coupling':self._coupling,
                                                   'linear':self._linear,
                                                   'input_tensor':self._input_tensor})
            
    @property
    def target_hamiltonian(self):
        return self._target_hamiltonian
    @target_hamiltonian.setter
    def target_hamiltonian(self,value):
        # we can add debugging terms here!
        self._target_hamiltonian=value
    @property
    def driver_hamiltonian(self):
        return self._driver_hamiltonian
    @driver_hamiltonian.setter
    def driver_hamiltonian(self,value):
        self._driver_hamiltonian=value

    def __get_basis_states(self):
        """Return (2**n_qubits, n_qubits) array of 0/1 bitstrings.
           MSB-first: qubit 0 is the leftmost column — matches QuTiP tensor(basis(...)) ordering."""
        num = 2**self.nqubits
        # each row is the binary representation (MSB first)
        arr = np.array([list(np.binary_repr(i, width=self.nqubits)) for i in range(num)], dtype=int)
        return arr

    def get_driver_hamiltonian(self):
        if self.nqubits is None:
            raise ValueError("Number of qubits must be set before getting driver Hamiltonian.")
        ham_x=0.
        for i in range(self.nqubits):
            ham_x+=SpinOperator([('x',i)],coupling=[1.0/self.nqubits],size=self.nqubits,verbose=0).qutip_op
        self._driver_hamiltonian=ham_x

    
    
    def set_quantum_annealing_parameters(self,time_steps,total_time,schedule_type='linear'):
        if self.target_hamiltonian is None:
            raise ValueError("Target Hamiltonian must be set before setting quantum annealing.")
        if self.driver_hamiltonian is None:
            self.get_driver_hamiltonian()
        
        self._time_steps=time_steps
        self._total_time=total_time
        self.time_list = np.linspace(0, total_time, time_steps)
        
        if schedule_type=='linear':
            self._driver_schedule=1-self.time_list/total_time
            self._target_schedule=self.time_list/total_time
        elif schedule_type=='quadratic':
            self._driver_schedule=(1-self.time_list/total_time)**2
            self._target_schedule=(self.time_list/total_time)**2
        else:
            raise ValueError("Unsupported schedule type. Use 'linear' or 'quadratic'.")
    
    
    
    @property
    def time_steps(self):
        return self._time_steps
    @property
    def total_time(self):
        return self._total_time
    @time_steps.setter
    def time_steps(self,value:int):
        if value<=0:
            raise ValueError("Time steps must be positive.")
        self._time_steps=value
    @total_time.setter
    def total_time(self,value:float):
        if value<=0:
            raise ValueError("Total time must be positive.")
        self._total_time=value    
    
    @property
    def driver_schedule(self):
        return self._driver_schedule
    @driver_schedule.setter
    def driver_schedule(self,value:np.ndarray):
        if value.shape[0]!=self.time_steps:
            raise ValueError("Driver schedule must be of shape (time_steps,).")
        self._driver_schedule=value
    @property
    def target_schedule(self):
        return self._target_schedule
    @target_schedule.setter
    def target_schedule(self,value:np.ndarray):
        if value.shape[0]!=self.time_steps:
            raise ValueError("Target schedule must be of shape (time_steps,).")
        self._target_schedule=value
    
    def quantum_annealing(self,nlevels:Optional[int]=None):
        
        if self.target_hamiltonian is None or self.driver_hamiltonian is None:
            raise ValueError("Both target and driver Hamiltonians must be set before quantum annealing.")
        if not hasattr(self,'time_list'):
            raise ValueError("Quantum annealing parameters must be set before quantum annealing.")
        
        # dumbest way to get the initial state
        # |-> state for one qubit
        minus = (basis(2, 0) - basis(2, 1)).unit()

        # Tensor product to get |+>^{⊗n}
        psi0 = tensor([minus for _ in range(self.nqubits)])
        # Convert to np.array
        psi0 = np.array(psi0.full()).flatten()

        
        
        if nlevels is not(None):
            spectrum=np.zeros((self.time_steps,nlevels))
            probs=np.zeros((self.time_steps,nlevels))
            fidelity=np.zeros((self.time_steps))
            e_t=np.zeros((self.time_steps))
        
        tbar=tqdm(enumerate(self.time_list))
        for r,t in tbar:
            tbar.set_description(f"Time {t:.2f}/{self.total_time:.2f}")
            H_t = self.driver_schedule[r]*self.driver_hamiltonian + self.target_schedule[r] * self.target_hamiltonian
            dt = self.time_list[1] - self.time_list[0]  # assuming
            if r == 0:
                psi = psi0[:]  # initial state: ground state of transverse field
            # Evolve state by small time step dt using matrix exponential
            psi = expm_multiply(-1j * H_t.data_as('csr_matrix') * dt, psi)
            
            if nlevels is not(None):
                energies_t,psi_levels=eigsh(H_t.data_as('csr_matrix'),k=nlevels,which='SA')
                spectrum[r,:]=energies_t
                probs[r,:]=np.abs(psi.conj().T @ psi_levels)**2
                e_t[r]=psi.conj().T @ (H_t.data_as('csr_matrix') @ psi)
            
        self._psi_annealing=psi
        
        if nlevels is not(None):
            return spectrum,probs,fidelity,e_t
        else:
            return None
    @property
    def psi_annealing(self):
        if not hasattr(self,'_psi_annealing'):
            raise ValueError("Quantum annealing must be performed before accessing psi_annealing.")
        return self._psi_annealing
    @psi_annealing.setter
    def psi_annealing(self,value):
        if value.shape[0]!=2**self.nqubits:
            raise ValueError("Psi annealing must be of shape (2**nqubits,).")
        self._psi_annealing=value
    
    def spectrum_analysis(self):
        if self.target_hamiltonian is None:
            raise ValueError("Target Hamiltonian must be set before spectrum analysis.")
        
        self._energies = self.target_hamiltonian.diag().real
        self._idx = np.argsort(self._energies)
        
    @property
    def energies(self):
        if not hasattr(self,'_energies'):
            raise ValueError("Spectrum analysis must be performed before accessing energies.")
        return self._energies[self._idx]
    @property
    def idx(self):
        if not hasattr(self,'_idx'):
            raise ValueError("Spectrum analysis must be performed before accessing idx.")
        return self._idx
    
    def get_p_gs(self):
        if not hasattr(self,'_psi_annealing'):
            raise ValueError("Quantum annealing must be performed before accessing p_gs.")
        if not hasattr(self,'_energies'):
            raise ValueError("Spectrum analysis must be performed before accessing p_gs.")
        
        # ground state index
        gs_index = self.idx[0]
        # probability of measuring the ground state
        p_gs = np.abs(self._psi_annealing[gs_index])**2
        return p_gs
    
    def get_minimum_gap(self):
        
        if self.target_hamiltonian is None or self.driver_hamiltonian is None:
            raise ValueError("Both target and driver Hamiltonians must be set before quantum annealing.")
        if not hasattr(self,'time_list'):
            raise ValueError("Quantum annealing parameters must be set before quantum annealing.")
        

        gap_list=[]
        
        tbar=tqdm(enumerate(self.time_list))
        for r,t in tbar:
            tbar.set_description(f"Time {t:.2f}/{self.total_time:.2f}")
            H_t = self.driver_schedule[r]*self.driver_hamiltonian + self.target_schedule[r] * self.target_hamiltonian
            energies_t=eigsh(H_t.data_as('csr_matrix'),k=2,which='SA',return_eigenvectors=False,maxiter=10000)
            gap_list.append(np.abs(energies_t[1]-energies_t[0])) # they are the two smallest eigenvalues but the order is unclear
        
        gap_array=np.array(gap_list)
        min_gap=np.min(gap_array)
        min_gap_time=self.time_list[np.argmin(gap_array)]
        
        self._min_gap=min_gap
        self._min_gap_time=min_gap_time
        
        return min_gap,min_gap_time/self.total_time
    
    @property
    def min_gap(self):
        if not hasattr(self,'_min_gap'):
            raise ValueError("Minimum gap must be computed before accessing min_gap.")
        return self._min_gap
    @property
    def min_gap_time(self):
        if not hasattr(self,'_min_gap_time'):
            raise ValueError("Minimum gap time must be computed before accessing min_gap_time.")
        return self._min_gap_time
        