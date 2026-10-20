"""What one measurement is worth, and how many of them you need. No cryo-EM inside: a rung hands over
an array of log-ratios and gets back a number.

    information   a per-particle log-ratio read as an amount of information
    power         that information read as a particle count, the deliverable
    composition   what fraction of a pool is state A, three estimators
    convergence   a rung measured on a bank ladder, extrapolated to an infinitely fine one

Nothing is re-exported -- every caller reaches a module by path.
"""
