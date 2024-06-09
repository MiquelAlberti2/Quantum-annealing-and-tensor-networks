import random
import numpy as np
from ncon import ncon
from collections import defaultdict

'''
Perfect Sampling with Unitary Tensor Networks (Andrew J. Ferris, Guifre Vidal)
https://arxiv.org/abs/1201.3974
'''

class SamplingMPS:

    def _density_matrix(self,mps: np.array, r_site: int) -> np.array:

        tensor_right = self.tensor_right[r_site + 1]

        return ncon([mps[0], np.conj(mps[0]), tensor_right], [[1, -1, 2], [1, -2, 3], [2, 3]])

    def _obtain_state_i(self, mps: np.array, r_site: int) -> int:

        reduce_matrix = self._density_matrix(mps=mps, r_site=r_site)

        pb_values = list(np.real(np.diag(reduce_matrix)))

        return (0, pb_values[0]) if random.uniform(0, 1) < pb_values[0] else (1, pb_values[1])

    def _proyecting_state(self, tensor: np.array, state: int) -> np.array:

        state_final = np.array([[1, 0]]) if state == 0 else np.array([[0, 1]])

        return ncon([state_final, tensor], [[-1, 1], [1, -2]])

    def _proyect_state_i(self, mps: np.array, r_site: int) -> np.array:

        if len(mps) > 1:

            mps_new = mps.copy()
            mps_new = mps_new[1:]

        else:
            mps_new = mps.copy()

        state_qubit, pb_state = self._obtain_state_i(mps=mps, r_site=r_site)

        state_to_proyect = mps[0].reshape(2, mps[0].shape[2])

        state_proyected = self._proyecting_state(tensor=state_to_proyect, state=state_qubit)

        state_proyected = state_proyected.reshape(1, mps[0].shape[2])

        mps_new[0] = ncon([state_proyected, mps_new[0]], [[-1, 1], [1, -2, -3]])

        mps_new[0] = 1 / np.sqrt(pb_state) * mps_new[0].reshape(1, 2, mps_new[0].shape[2])

        return mps_new, state_qubit

    def _one_sample(self, mps: np.array) -> np.array:

        mps_sample = mps.copy()

        sample = []

        for i in range(len(mps_sample)):

            mps_sample, state_qubit = self._proyect_state_i(mps=mps_sample, r_site=i)

            sample.append(state_qubit)

        return sample

    def _right_envs(self, mps: np.array):

        ortho_renv = [None for _ in range(len(mps))]

        # contracts with the conjugate
        ortho_renv[-1] = ncon([mps[-1], np.conj(mps[-1])], [[-1, 1, 2], [-2, 1, 2]])

        for s in range(len(mps) - 2, -1, -1):

            ortho_renv[s] = ncon(
                [mps[s], np.conj(mps[s]), ortho_renv[s + 1]],
                [[-1, 1, 2], [-2, 1, 3], [2, 3]],
            )

        ortho_renv.append(np.eye(1)) # np.eye(1) = [[1.]]

        return ortho_renv

    def sampling(self, mps: np.array, n_samples: int):

        self.tensor_right = self._right_envs(mps=mps)

        list_samples = [self._one_sample(mps=mps) for _ in range(n_samples)]

        counting = defaultdict(int)

        for sub_list in list_samples:
            key = "".join(map(str, sub_list))
            counting[key] += 1

        return {key: count / n_samples for key, count in counting.items()}