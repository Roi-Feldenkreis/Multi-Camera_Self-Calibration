import numpy as np


def mycorr2(X, G, Gn, Gn2):
    """
    mycorr2 computes the 2D correlation for use with im2col and col2im.
    
    Parameters:
    X   : np.ndarray
        Matrix of data (each column can represent a patch).
    G   : np.ndarray
        Reference vector to correlate against.
    Gn  : np.ndarray
        G - mean(G), precomputed to save time.
    Gn2 : float
        Square root of the sum of squares of Gn, also precomputed.
    
    Returns:
    R   : np.ndarray
        The correlation result for each column of X.
    """
    # Step 1: Compute the mean of each column in X and subtract it to center the data
    mX = np.mean(X, axis=0, keepdims=True)  # Mean for each column
    mXn = X - mX  # X with mean subtracted
    
    # Step 2: Compute the sum of squares for each column
    smX = np.sum(mXn**2, axis=0)  # Sum of squared deviations
    
    # Step 3: Compute the numerator (mXn.T @ Gn)
    numerator = np.dot(mXn.T, Gn).T  # Equivalent to (mXn' * Gn)' in MATLAB
    
    # Step 4: Compute the denominator (smX * Gn2)
    denominator = smX * Gn2
    
    # Step 5: Compute the correlation by element-wise division
    R = numerator / denominator
    
    return R

# Example data
"""
X = np.random.rand(100, 10)  # 100x10 data matrix
G = np.random.rand(100)      # Reference vector

# Precompute Gn and Gn2
Gn = G - np.mean(G)
Gn2 = np.sqrt(np.sum(Gn**2))

# Run the correlation function
R = mycorr2(X, G, Gn, Gn2)

print(R)  # Output: Correlation results
"""