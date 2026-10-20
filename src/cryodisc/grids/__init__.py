"""The sampling grids every module shares, before any molecule is placed on them.

    fourier   the unshifted fft grid, and the whitening and shift conventions that ride on it
    poses     the icosahedral pose bank over SO(3), and the shift lattice

Both are pure quadrature: a set of nodes and the weights to sum over them, with no cryo-EM inside.
Nothing is re-exported -- every caller reaches a module by path.
"""
