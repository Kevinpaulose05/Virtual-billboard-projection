import numpy as np
from est_homography import est_homography


def warp_pts(X, Y, interior_pts):
    
    # Function call to estimate the homography matrix
    H = est_homography(X, Y)
    
    # Initialize an array to store the warped points
    warped_pts = np.zeros_like(interior_pts)
    
    # Obtaining the number of interior points
    n = interior_pts.shape[0]

    # Create a new column of ones for homogeneous coordinates
    last_column = np.ones(n).reshape(-1,1)
    
    # Adding the homogeneous coordinates to the interior_pts
    homo_coord = np.concatenate((interior_pts, last_column), axis= 1)
        
    # Apply homography transformation to the points
    warped_pts_homo = H @ homo_coord.T
    
    # Normalize the coordinates by dividing by the last row (homogeneous coordinate)
    warped_pts_homo = warped_pts_homo/warped_pts_homo[-1]
    
    # Transpose the warped points back to (x, y, 1) format
    warped_pts_homo_norm = warped_pts_homo.T
    
    # Remove the last column (homogeneous coordinate) to obtain the final warped points
    warped_pts = np.delete(warped_pts_homo_norm, 2, 1)

    return warped_pts
