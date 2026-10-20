"""The differentiable forward model: a state in, templates out.

    cryojax_model   renders either state of a pair into a bank of templates
    optics          `CTFParams`, the per-particle 2D CTF cryoJAX renders through
    expansion       builds seven of every eight templates from the eighth by index permutation

Nothing is re-exported; every caller reaches a module by path.
"""
