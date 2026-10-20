"""Particle stacks: simulated with a known answer, or read from real cryo-EM files.

sim        a stack you made, with the ground truth kept beside it
metadata   a real stack's parameters, from cryoSPARC .cs or RELION .star
images     the pixels those rows point at
ctffit     that stack's CTF, remeasured rather than trusted
em         sigma^2(k), the template contrast and the occupancies, fitted from either
snr        the one whitened-power scalar every rung of a pool lives on
"""
