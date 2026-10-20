"""The orchestration layer: the two entry points, end to end.

    chain            predict how separable two states are, and how many particles it takes
    fraction         estimate the fraction of each state in a real, mixed pool
    operating_point  what a ladder was measured at, and when two of them may be divided

Nothing is re-exported -- every caller reaches a module by path.
"""
