# The code is written by Jalil Nourmohammadi Khiarak and all copy rights is reserved.

import unittest
import numpy as np
from ..models.kmean_plus_plus import kmeans_plus_plus_init, kmeans

class TestKMeans(unittest.TestCase):
    def setUp(self):
        # Create a simple, clearly separated dataset
        self.X = np.array([
            [1, 2], [1, 4], [1, 0],  # Cluster 1 (near [1, 2])
            [10, 2], [10, 4], [10, 0] # Cluster 2 (near [10, 2])
        ])
        self.k = 2

    def test_initialization_shape(self):
        """Verify k-means++ returns the correct number of centroids."""
        centroids = kmeans_plus_plus_init(self.X, self.k)
        self.assertEqual(centroids.shape, (self.k, self.X.shape[1]))

    def test_initialization_membership(self):
        """Ensure initial centroids are chosen from the original dataset."""
        centroids = kmeans_plus_plus_init(self.X, self.k)
        for c in centroids:
            # Check if each centroid exists in X
            self.assertTrue(any(np.allclose(c, x) for x in self.X))

    def test_clustering_convergence(self):
        """Verify the final output has correct labels and centroid counts."""
        centers, labels = kmeans(self.X, self.k)
        
        # Check output shapes
        self.assertEqual(len(labels), len(self.X))
        self.assertEqual(len(centers), self.k)
        
        # In this specific dataset, points should be split 3 and 3
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.assertEqual(list(counts), [3, 3])

    def test_reproducibility_with_seed(self):
        """Check that setting a seed (via np.random) produces consistent results."""
        np.random.seed(42)
        centers1, _ = kmeans(self.X, self.k)
        
        np.random.seed(42)
        centers2, _ = kmeans(self.X, self.k)
        
        np.testing.assert_array_almost_equal(centers1, centers2)

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
