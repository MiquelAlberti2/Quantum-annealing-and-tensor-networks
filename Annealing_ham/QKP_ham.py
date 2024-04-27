import dimod
from neal import SimulatedAnnealingSampler

class QKP_Hamiltonian:
    def __init__(self, N, values, weights, W_capacity):
        self.N = N
        self.values = values
        self.weights = weights
        self.W_capacity = W_capacity

        # Define cost function with binary variables
        x = [dimod.Binary(f'x_{i}') for i in range(N)]

        cost = - dimod.quicksum(values[i][j] * x[i] * x[j] for i in range(N) for j in range (i+1)) # values
        cost -= 0.9603*(W_capacity - dimod.quicksum(weights[i] * x[i] for i in range(N)))
        cost += 0.0371*(W_capacity - dimod.quicksum(weights[i] * x[i] for i in range(N)))**2

        # Convert to Ising model (spin variables)
        self.model = dimod.BinaryQuadraticModel(cost.linear, cost.quadratic, cost.offset, vartype='BINARY')
        self.H_dict = self.model.to_ising()

    def run_simulated_annealing(self):
        sampleset = SimulatedAnnealingSampler().sample(self.model, num_reads=1000)
        return sampleset.first.sample

    def convert_to_MPO(self):
        pass