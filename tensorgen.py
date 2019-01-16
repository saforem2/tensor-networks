"""
tensorgen.py

Original author: Judah Unmuth-Yockey
"""
from pprint import pprint

import numpy as np
import o3funcs as o3f


# pylint:disable=invalid-name
def tensorgen(beta, D, lmax):
    """
    Generates the initial tensor for O(3) spin.

    Parameters
    ----------
    beta : the inverse temperature
    D :    the absolute value of the largest `m' quantum number.
           A reasonable assumption is that D <= lmax.
    lmax : the largest total angular momentum value

    Returns
    -------
    top_tensor : this is a tensor parameterized in terms of it's
                 left, top, and right legs.
    bot_tensor : this is a tensor parameterized in terms of it's
                 right, bottom, and left legs.

    Notes
    -----
    - these tensors are parameterized in terms of their `m' quantum numbers
      and there are two of them.
    - Each one handles a different direction with the legs that parameterize
      the tensor.
    - The orientation on the lattice is left, right, up, down.
    - The top tensor is parameterized left, up, right.
    - The bottom tensor is parameterized right, down, left.

    """
    weights = [o3f.Acoeff(a, beta) for a in range(lmax+1)]

    S = {}
    for M in range(-2*D, 2*D+1):
        for m1 in range(max([-D, M-D]), min([D, M+D])+1):
            S[(m1, M)] = np.zeros((lmax+1, lmax+1, 2*lmax+1))
            for l1 in range(abs(m1), lmax+1):
                for l2 in range(abs(M-m1), lmax+1):
                    for L in range(abs(M), l1+l2+1):
                        try:
                            S[(m1, M)][l1, l2, L] += (
                                np.sqrt(weights[l1]
                                        * weights[l2]
                                        * (2*l1+1)*(2*l2+1)
                                        / (4*np.pi*(2*L+1)))
                                * o3f.cleb_gor(l1, l2, L, m1, M-m1, M)
                                *o3f.cleb_gor(l1, l2, L, 0, 0, 0)
                            )
                        except KeyError:
                            S[(m1, M)][l1, l2, L] = (
                                np.sqrt(weights[l1]
                                        * weights[l2]
                                        * (2*l1+1)
                                        * (2*l2+1)
                                        / (4*np.pi*(2*L+1)))
                                * o3f.cleb_gor(l1, l2, L, m1, M-m1, M)
                                * o3f.cleb_gor(l1, l2, L, 0, 0, 0)
                            )

    Ttop = {}
    Tbot = {}
    for M in range(-2*D, 2*D+1):
        for m1 in range(max([-D, M-D]), min([D, M+D])+1):
            for m3 in range(max([-D, M-D]), min([D, M+D])+1):
                Ttop[(m1, M-m1, m3)] = (
                    np.einsum('ika, jla', S[(m1, M)], S[(m3, M)])
                )
                Tbot[(m3, M-m3, m1)] = Ttop[(m1, M-m1, m3)]

    return Ttop, Tbot

beta = 0.5  # the inverse temperature
lmax = 1    # the largest l value
D = lmax       # the absolute value of the largest m value

top, bot = tensorgen(beta, D, lmax)

pprint(top)
print("=====================================================================")
pprint(bot)
