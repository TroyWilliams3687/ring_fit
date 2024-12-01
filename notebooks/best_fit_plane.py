
#!/usr/bin/env python3
#-*- coding:utf-8 -*-

"""
This modules contains the equations and code for the equations to calculate the 
elastic modulus of rocks based on input data.

The equation list can be found here: 
- http://subsurfwiki.org/wiki/Young%27s_modulus 

Copyright (c) 2019 Troy Williams

License: The MIT License (http://www.opensource.org/licenses/mit-license.php)
"""

#Constants
__uuid__ = ''
__author__ = 'Troy Williams'
__email__ = 'troy.williams@bluebill.net'
__copyright__ = 'Copyright (c) 2019, Troy Williams'
__date__ = '2019-07-19'
__maintainer__ = 'Troy Williams'

# import standard modules
import numpy as np
import scipy.linalg

def A_matrix(c1, c2):
    """
    c1 - np.ndarray - first column of matrix
    c2 - np.ndarray - second column of matrix
    
    Returns
    A matrix with columns |c1, c2, identity|
    
    """
    
    return np.c_[c1, c2, np.ones(c1.shape[0])]

def best_fit_plane(X, Y, Z):
    """
    Takes a series of 3D points and determines the best fit plane of the form
    
    ax + by + cz + d =0
    
    Returns
    -------
    
    The coefficients of the plane: a,b,c,d 
    
    """
    
    # for this to work properly we need to determine which plane to attempt a 
    # fit with. We can do this be choosing the largest determinant from 
    # det |A^T A|
    
    A_XY = A_matrix(X, Y)
    b_XY = -1*Z

    A_XZ = A_matrix(X, Z)
    b_XZ = -1*Y

    A_YZ = A_matrix(Y, Z)
    b_YZ = -1*X

    # take the determinant of the A^T * A matrix multiplication (order counts)
    # and figure out which is the largest one and use that plane to perform
    # the regression with
    dets = [np.linalg.det(np.matmul(np.transpose(A),A)) for A in (A_XY, A_XZ, A_YZ)]
    largest = dets.index(max(dets))

    a = None
    b = None
    c = None
    d = None
    plane = None
    
    if largest == 0:
        plane = 'XY'
        x, _, _, _ = scipy.linalg.lstsq(A_XY, b_XY)

        c = 1
        a, b, d = x           
        
    elif largest == 1:
        plane = 'XZ'
        x, _, _, _ = scipy.linalg.lstsq(A_XZ, b_XZ)

        b = 1
        a, c, d = x    

    elif largest == 2:
        plane = 'YZ'
        x, _, _, _ = scipy.linalg.lstsq(A_YZ, b_YZ)

        a = 1
        b, c, d = x    
        
    else:
        raise ValueError("Something went wrong...")
    
    
    # normalize the vector
    m = np.sqrt(a*a + b*b + c*c)
    a = a/m
    b = b/m
    c = c/m

    return a, b, c, d, plane
    
