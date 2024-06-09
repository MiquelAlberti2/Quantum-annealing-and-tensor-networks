from overrides import override
from neal import SimulatedAnnealingSampler

from solver import Solver

class Neal_annealing_solver(Solver):
    def __init__(self, W, wt, val):
        super().__init__(W, wt, val)

    @override
    def run(self, time = 1000):
        '''
        Runs simulated annealing using D-Wave's Neal library

        time: number of reads
        '''
        sampleset = SimulatedAnnealingSampler().sample(self.model, num_reads = time)
        self.solution_items = []
        for key in sampleset.first.sample:
            if sampleset.first.sample[key] == 1:
                self.solution_items.append(key)

        return self.solution_items
    
    