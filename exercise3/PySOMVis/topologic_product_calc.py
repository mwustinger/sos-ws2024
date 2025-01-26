import numpy as np
from functools import reduce

class TopologicProductCalculator():
    def __init__(self, som):
        self.som = som
        self.w = self._get_weight_array()
        self.e = self._get_euclidean_array()
        
        self.cols = self.w.shape[0] # x-axis
        self.rows = self.w.shape[1] # y-axis
        self.N = self.cols * self.rows

        # Precompute k-nearest neighbors
        self.nIs = np.array([[self._calc_k_nearest_neighbors(self.w, x, y) for y in range(self.rows)] for x in range(self.cols)])
        self.nOs = np.array([[self._calc_k_nearest_neighbors(self.e, x, y) for y in range(self.rows)] for x in range(self.cols)])

        # Precompute Q values
        self.Q1s = np.array([[self._calc_Q(self.w, x, y) for y in range(self.rows)] for x in range(self.cols)])
        self.Q2s = np.array([[self._calc_Q(self.e, x, y) for y in range(self.rows)] for x in range(self.cols)])

    def _get_weight_array(self):
        arr = np.array(self.som.get_weights())
        return np.swapaxes(arr, 0, 1)

    def _get_euclidean_array(self):
        xx, yy = self.som.get_euclidean_coordinates()
        arr = np.stack((xx, yy), axis=-1)
        return np.swapaxes(arr, 0, 1)

    def _calc_k_nearest_neighbors(self, weights, xj, yj):
        wj = weights[xj, yj]  # Get the weight at (xj, yj)
        dist = np.linalg.norm(weights - wj, axis=-1)  # Broadcasted Euclidean distance
        sorted_indices = np.argsort(dist, axis=None)
        coordinates = np.unravel_index(sorted_indices, dist.shape)
        nn = np.column_stack(coordinates)
        return nn

    def _calc_Q(self, weights, xj, yj):
        wj = weights[xj, yj]
        nOsj = self.nOs[xj, yj]
        nIsj = self.nIs[xj, yj]
        dOs = np.linalg.norm(weights[nOsj[:, 0], nOsj[:, 1]] - wj, axis=-1)
        dIs = np.linalg.norm(weights[nIsj[:, 0], nIsj[:, 1]] - wj, axis=-1)
        q = np.divide(dIs, dOs, where=dOs != 0)  # Safe division, avoid division by zero
        return q

    def _calc_P(self, q, k):
        product = np.prod(q[1:k+1])
        return product ** (1 / k)
        
    def k_nearest_neighbor_input(self, xj, yj, k=None):
        nIsj = self.nIs[xj, yj]
        return nIsj if k is None else nIsj[k]
    
    def k_nearest_neighbor_output(self, xj, yj, k=None):
        nOsj = self.nOs[xj, yj]
        return nOsj if k is None else nOsj[k]
    
    def Q1(self, xj, yj, k=None):
        Q1sj = self.Q1s[xj, yj]
        return Q1sj if k is None else Q1sj[k]
    
    def Q2(self, xj, yj, k=None):
        Q2sj = self.Q2s[xj, yj]
        return Q2sj if k is None else Q2sj[k]
    
    def P1(self, xj, yj, k):
        return self._calc_P(self.Q1s[xj, yj], k)
    
    def P2(self, xj, yj, k):
        return self._calc_P(self.Q2s[xj, yj], k)
    
    def P3(self, xj, yj, k):
        product = np.prod(self.Q1s[xj, yj][1:k+1]) * np.prod(self.Q2s[xj, yj][1:k+1])
        return product ** (1 / (2*k))
    
    # Disclaimer: This function is computationally very expensive
    def P(self):
        grid_indices = np.array(list(np.ndindex(self.cols, self.rows)))
        k_values = np.arange(1, self.N)
        log_P3_values = np.array([
            np.log(self.P3(x, y, k)) for x, y in grid_indices for k in k_values
        ])
        total_sum = np.sum(log_P3_values)
        return (1 / (self.N * (self.N - 1))) * total_sum