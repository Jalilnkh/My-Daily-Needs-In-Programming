# The code is written by Jalil Nourmohammadi Khiarak and all copy rights is reserved.

import numpy as np

def kmeans_plus_plus_init(X, k):
    """
    Initializes centroids using the K-means++ algorithm.
    """
    n_samples, n_features = X.shape
    # 1. Randomly pick the first centroid
    centroids = [X[np.random.randint(n_samples)]]

    for _ in range(1, k):
        # 2. Compute squared distances to the nearest centroid for all points
        # Using [Scipy's cdist](https://docs.scipy.org) is common, 
        # but here is the manual broadcast version:
        distances = np.array([min([np.inner(c-x, c-x) for c in centroids]) for x in X])
        
        # 3. Probabilistic selection: P(x) is proportional to D(x)^2
        probs = distances / distances.sum()
        cumulative_probs = probs.cumsum()
        r = np.random.rand()
        
        # Select the data point where the random value falls in the CDF
        for i, threshold in enumerate(cumulative_probs):
            if r < threshold:
                centroids.append(X[i])
                break
                
    return np.array(centroids)

def kmeans(X, k, max_iters=100):
    # Step 1: Smart Initialization
    centroids = kmeans_plus_plus_init(X, k)
    
    for _ in range(max_iters):
        # Step 2: Assign clusters (Nearest Neighbor)
        # We use [Numpy's linalg.norm](https://numpy.org) for distance
        labels = np.array([np.argmin([np.linalg.norm(x - c) for c in centroids]) for x in X])
        
        # Step 3: Update centroids (Mean of points in cluster)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    return centroids, labels

# Quick Test
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    centers, labels = kmeans(X, k=4)
    print("Final Centroids:\n", centers)
