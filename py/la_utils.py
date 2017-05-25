def GaussJordanElimination(m, eps =1.0 / (10 ** 10)):
    """
    Modifies (inplace) given matrix into reduced row echelon form, so it becomes partitioned into two blocks, 
    with the identity matrix in the top-left block.
    """
    nrows, ncols = m.shape[0:2]
    ident_size = min(nrows, ncols) # size of the top-left identity block
    for i_diag in range(0, ident_size):
        # Find max pivot
        maxrow = i_diag
        pivot = abs(m[i_diag][i_diag])
        for row in range(i_diag + 1, nrows):
            pivot_cand = abs(m[row][i_diag])
            if pivot_cand > pivot:
                maxrow = row
                pivot = pivot_cand

        if pivot < eps:  # singular?
            return False

        pivot_inv = 1/pivot

        # swap rows
        if maxrow != i_diag:
            for col in range(i_diag, ncols):
                m[i_diag,col], m[maxrow,col] = m[maxrow,col], m[i_diag,col]

        # eliminate column y
        for row in range(0, nrows):
            if row == i_diag:
                # scale row so that the diagonal element becomes unity
                diag_inv = 1 / m[i_diag][i_diag]
                for col in range(i_diag, ncols):
                    m[i_diag][col] *= diag_inv
            else:
                c = m[row][i_diag] / m[i_diag][i_diag]
                for col in range(i_diag, ncols):
                    m[row][col] -= m[i_diag][col] * c

    return True