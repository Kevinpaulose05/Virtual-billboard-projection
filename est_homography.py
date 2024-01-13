import numpy as np


def est_homography(X, Y):

    # Define coordinates of goal corners (X) and the Penn logo corners (Y)    
    X1, X2, X3, X4 = X[0, 0], X[1, 0], X[2, 0], X[3, 0]
    Y1, Y2, Y3, Y4 = X[0, 1], X[1, 1], X[2, 1], X[3, 1]
    Xd1, Xd2, Xd3, Xd4 = Y[0, 0], Y[1, 0], Y[2, 0], Y[3, 0]
    Yd1, Yd2, Yd3, Yd4 = Y[0, 1], Y[1, 1], Y[2, 1], Y[3, 1]

    # Build 8x9 matrix A for direct linear tranformation
    A = np.array([[-X1, -Y1, -1, 0, 0, 0, X1 * Xd1, Y1 * Xd1, Xd1],
                  [0, 0, 0, -X1, -Y1, -1, X1 * Yd1, Y1 * Yd1, Yd1],
                  [-X2, -Y2, -1, 0, 0, 0, X2 * Xd2, Y2 * Xd2, Xd2],
                  [0, 0, 0, -X2, -Y2, -1, X2 * Yd2, Y2 * Yd2, Yd2],
                  [-X3, -Y3, -1, 0, 0, 0, X3 * Xd3, Y3 * Xd3, Xd3],
                  [0, 0, 0, -X3, -Y3, -1, X3 * Yd3, Y3 * Yd3, Yd3],
                  [-X4, -Y4, -1, 0, 0, 0, X4 * Xd4, Y4 * Xd4, Xd4],
                  [0, 0, 0, -X4, -Y4, -1, X4 * Yd4, Y4 * Yd4, Yd4]])

    # Perform SVD on A
    u, s, vh = np.linalg.svd(A)
    
    
    # Initialize an empty list to store the elements of the homography matrix
    h = []
    
    # Last column of vh contains the elements of the homography matrix
    for j in range(9):
        a = vh[8, j]
        h.append(a)
    # h = np.array(h)
    
    # Reshape h into a 3x3 matrix
    H = np.reshape(h, (3, 3))


    return H
