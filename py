import numpy as np

def tri3d_membrane_stiffness(X, E, nu, t):
    """
    3D triangular membrane element (CST) stiffness matrix.

    Parameters
    ----------
    X : (3, 3) array
        Nodal coordinates [[x1,y1,z1],
                           [x2,y2,z2],
                           [x3,y3,z3]]
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    t : float
        Thickness

    Returns
    -------
    K_global : (9, 9) array
        Global stiffness matrix for translational DOFs:
        [ux1,uy1,uz1, ux2,uy2,uz2, ux3,uy3,uz3]
    """

    X = np.asarray(X, dtype=float)
    X1, X2, X3 = X

    # --- Step 1: local basis e1, e2, e3 ---
    v1 = X2 - X1
    v2 = X3 - X1

    norm_v1 = np.linalg.norm(v1)
    if norm_v1 < 1e-12:
        raise ValueError("Degenerate element: nodes 1 and 2 are coincident.")

    e1 = v1 / norm_v1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-12:
        raise ValueError("Degenerate element: area ~ 0, nodes are collinear.")

    e3 = n / norm_n
    e2 = np.cross(e3, e1)

    R = np.column_stack((e1, e2, e3))  # 3x3

    # --- Step 2: local coordinates ---
    xloc = (X - X1) @ R  # (3,3); rows are nodes; columns are local x,y,z
    x1, y1 = xloc[0, 0], xloc[0, 1]
    x2, y2 = xloc[1, 0], xloc[1, 1]
    x3, y3 = xloc[2, 0], xloc[2, 1]

    # --- Step 3: area and b,c coefficients ---
    A = 0.5 * np.linalg.det(np.array([
        [1.0, x1, y1],
        [1.0, x2, y2],
        [1.0, x3, y3]
    ]))

    if abs(A) < 1e-12:
        raise ValueError("Degenerate element: zero area.")

    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2

    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    B2D = (1.0 / (2.0 * A)) * np.array([
        [b1, 0.0, b2, 0.0, b3, 0.0],
        [0.0, c1, 0.0, c2, 0.0, c3],
        [c1,  b1, c2,  b2, c3,  b3]
    ])

    # --- Step 4: plane-stress material matrix ---
    coeff = E / (1.0 - nu**2)
    D = coeff * np.array([
        [1.0,  nu,  0.0],
        [nu,  1.0,  0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])

    # --- Step 5: local in-plane stiffness (6x6) ---
    K_loc2D = t * A * (B2D.T @ D @ B2D)

    # --- Step 6: transformation to global (9x9) ---
    # Local->global mapping for each node: [u,v]^T = T_i * [ux,uy,uz]^T
    # T_i is (2x3): rows are e1^T and e2^T.
    Tnode = np.vstack((e1, e2))  # 2x3

    # Build block-diagonal T (6x9)
    T = np.zeros((6, 9))
    # node 1
    T[0:2, 0:3] = Tnode
    # node 2
    T[2:4, 3:6] = Tnode
    # node 3
    T[4:6, 6:9] = Tnode

    # Global stiffness
    K_global = T.T @ K_loc2D @ T  # (9x9)

    return K_global
