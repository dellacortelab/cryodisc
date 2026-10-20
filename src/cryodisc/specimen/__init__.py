"""One molecule in two conformations, as atoms and as a grid.

    structures   two deposited models in one frame and on one atom set
    density      that molecule as a GRID instead: rendered, banded, resampled, put in a frame

`structures` is the atomic truth the templates render from; `density` is the same pair voxelized
where the templates live. Nothing is re-exported -- every caller reaches a module by path.
"""
