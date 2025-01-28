import unittest
import numpy as np
from topologic_product_calc import TopologicProductCalculator

class MockSOM:
    def __init__(self):
        self.weights = np.array([
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
            [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]],
            [[1.3, 1.4], [1.5, 1.6], [1.7, 1.8]],
        ])
        self.xx = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.yy = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])

    def get_weights(self):
        return self.weights

    def get_euclidean_coordinates(self):
        return self.xx, self.yy

class TestTopologicProductCalculator(unittest.TestCase):
    def setUp(self):
        self.mock_som = MockSOM()
        self.calculator = TopologicProductCalculator(self.mock_som)

    def test_k_nearest_neighbors_correct_length(self):
        weights = self.calculator._get_weight_array()
        neighbors = self.calculator._calc_k_nearest_neighbors(weights, 1, 1)
        self.assertEqual(len(neighbors), 9, "Number of neighbors should match total grid size.")

    def test_k_nearest_neighbors_sorted(self):
        weights = self.calculator._get_weight_array()
        neighbors = self.calculator._calc_k_nearest_neighbors(weights, 1, 1)
        distances = [
            np.linalg.norm(weights[n[0], n[1]] - weights[1, 1])
            for n in neighbors
        ]
        self.assertTrue(all(x <= y for x, y in zip(distances, distances[1:])), "Neighbors should be sorted by distance.")

    def test_k_nearest_neighbors_first_is_self(self):
        weights = self.calculator._get_weight_array()
        neighbors = self.calculator._calc_k_nearest_neighbors(weights, 1, 1)
        np.testing.assert_array_equal(neighbors[0], [1, 1], "The closest neighbor should be the point itself.")

    def test_calc_Q_values_non_negative(self):
        weights = self.calculator._get_weight_array()
        Q = self.calculator._calc_Q(weights, 1, 1)
        self.assertTrue(np.all(Q >= 0), "All Q values should be non-negative.")

    def test_calc_Q_matches_manual_calculation(self):
        weights = self.calculator._get_weight_array()
        wj = weights[1, 1]
        dOs = np.linalg.norm(weights[self.calculator.nOs[1, 1][:, 0],
                                     self.calculator.nOs[1, 1][:, 1]] - wj, axis=1)
        dIs = np.linalg.norm(weights[self.calculator.nIs[1, 1][:, 0],
                                     self.calculator.nIs[1, 1][:, 1]] - wj, axis=1)
        expected_Q = np.divide(dIs, dOs, where=dOs != 0)
        Q = self.calculator._calc_Q(weights, 1, 1)
        np.testing.assert_array_almost_equal(Q, expected_Q, err_msg="Q calculation does not match manual calculation.")

    def test_calc_Q_handles_zero_division(self):
        weights = self.calculator._get_weight_array()
        self.calculator.nOs[(1, 1)] = np.array([[1, 1]])  # Force zero division
        Q = self.calculator._calc_Q(weights, 1, 1)
        self.assertTrue(np.all(np.isfinite(Q)), "Q should handle zero division gracefully.")

    def test_calc_P_correctness(self):
        q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        k = 3
        result = self.calculator._calc_P(q, k)
        expected = (2 * 3 * 4) ** (1 / 3)
        self.assertAlmostEqual(result, expected, "P calculation does not match expected value.")

    def test_calc_P_handles_single_element(self):
        q = np.array([1.0])
        k = 0
        result = self.calculator._calc_P(q, k)
        self.assertAlmostEqual(result, 1.0, "P should handle single-element q correctly.")

    def test_P1_P2_P3_relationship(self):
        k = 3
        P1 = self.calculator.P1(1, 1, k)
        P2 = self.calculator.P2(1, 1, k)
        P3 = self.calculator.P3(1, 1, k)

        self.assertGreaterEqual(P1, 0, "P1 should be non-negative.")
        self.assertGreaterEqual(P2, 0, "P2 should be non-negative.")
        self.assertGreaterEqual(P3, 0, "P3 should be non-negative.")
        self.assertTrue(min(P1, P2) <= P3 <= max(P1, P2), "P3 should lie between P1 and P2.")

    def test_P3_symmetry(self):
        k = 3
        P3_a = self.calculator.P3(1, 1, k)
        P3_b = self.calculator.P3(1, 1, k)
        self.assertAlmostEqual(P3_a, P3_b, "P3 should be symmetric and consistent.")

