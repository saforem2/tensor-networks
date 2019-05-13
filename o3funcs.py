"""
o3funcs.py

Original Author: Judah Yockey

Current Author: Sam Foreman (github/twitter: @saforem2)
"""
#pylint:disable=invalid-name
import numpy as np
from scipy.misc import factorial
from scipy.special import iv
from math import exp
# np.set_printoptions(suppress=True, linewidth=200)
from itertools import product, combinations_with_replacement #, imap
from functools import reduce
from scipy.linalg import block_diag
from time import time
from collections import Counter
import operator
from pprint import pprint


prod = lambda factors: reduce(operator.mul, factors, 1)

list_add = lambda lol: list(sum(lol, []))

allcidl = lambda alist: list(set(list_add(alist)))

def Acoeff(n, x):
    """Acoeff n, x."""
    if ((x == 0) and (n != 0)):
        return 0.0
    if (x == 0) and (n == 0):
        return 4 * np.pi
    return (2 * np.pi) ** (1.5) * iv(n + 0.5, x) / np.sqrt(x)

# def Acoeff(n, x):
#     """Acoeff n, x."""
#     if ((x == 0) and (n != 0)):
#         return 0.0
#     if (n == 0):
#         return 1.
#     return iv(n + 0.5, x) / iv(0.5, x) 

def tensorgen(beta, mu, D, lmax):
    """
    Generates the initial tensor for O(3) spin.

    Parameters
    ----------
    beta : the inverse temperature
    mu : the chemical potential on the time links.
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
    weights = [Acoeff(a, beta) for a in range(lmax+1)]

    S = {}
    for M in range(-2*D, 2*D+1):
        for m1 in range(max([-D, M-D]), min([D, M+D])+1):
            S[(m1, M)] = np.zeros((lmax+1, lmax+1, 2*lmax+1))
            for l1 in range(abs(m1), lmax+1):
                for l2 in range(abs(M-m1), lmax+1):
                    for L in range(abs(M), l1+l2+1):
                        try:
                            S[(m1, M)][l1, l2, L] += (
                                np.sqrt(exp(mu*(M-m1))
                                        *weights[l1]*weights[l2]
                                        *(2*l1+1)*(2*l2+1)/(4*np.pi*(2*L+1)))
                                * cleb_gor(l1, l2, L, m1, M-m1, M)
                                * cleb_gor(l1, l2, L, 0, 0, 0)
                            )

                        except KeyError:
                            S[(m1, M)][l1, l2, L] = (
                                np.sqrt(exp(mu*(M-m1))
                                        *weights[l1]*weights[l2]
                                        *(2*l1+1)*(2*l2+1)/(4*np.pi*(2*L+1)))
                                * cleb_gor(l1, l2, L, m1, M-m1, M)
                                * cleb_gor(l1, l2, L, 0, 0, 0)
                            )

    Ttop = {}
    Tbot = {}
    for M in range(-2*D, 2*D+1):
        for m1 in range(max([-D, M-D]), min([D, M+D])+1):
            for m3 in range(max([-D, M-D]), min([D, M+D])+1):
                Ttop[(m1, M-m1, m3)] = np.einsum(
                    'ika, jla', S[(m1, M)], S[(m3, M)]
                )
                Tbot[(m3, M-m3, m1)] = Ttop[(m1, M-m1, m3)]



    return Ttop, Tbot

def cleb_gor(j1, j2, j, m1, m2, m):
    """
    Returns the Clebsch-Gordan coeffcient.

    Parameters
    ----------
    j1, j2, j : The total angular momenta input
    m1, m2, m : The z-components of angular momenta input

    Returns
    -------
    real_number : The numerical value of the Clebsch-Gordan coeffcient.

    Notes
    -------
    Note that in the sum NONE of the factorial arguments can have negative
    values.  Thus, zmin and zmax are there to make sure that the sums have a
    cut-off.

    """
    cond = (
        (m1 + m2 == m)
        and (abs(j1 - j2) <= j <= (j1 + j2))
        and (-j <= m <= j)
        and (-j1 <= m1 <= j1)
        and (-j2 <= m2 <= j2)
    )
    if cond:
        zmin = min([j1+j2-j, j1-m1, j2+m2])
        zmax = max([-j+j2-m1, -j+j1+m2, 0])

        temp = sum([((-1)**z /
                    (factorial(z) * factorial(j1+j2-j-z) *
                     factorial(j1-m1-z) * factorial(j2+m2-z) *
                     factorial(j-j2+m1+z) * factorial(j-j1-m2+z)))
                   for z in range(zmax, zmin+1)])

        triangle = np.sqrt(
            factorial(j1+j2-j) * factorial(j1-j2+j) * factorial(j2+j-j1)
            / factorial(j1+j2+j+1)
        )
        wanted =  (
            triangle * np.sqrt(factorial(j1+m1) * factorial(j1-m1)
                               * factorial(j2+m2) * factorial(j2-m2)
                               * factorial(j+m) * factorial(j-m)
                               * (2*j+1)) * temp
        )
        return wanted

    else:
        return 0.0


# def CG_zero_mom_proj(j1, j2, j3):
#     g = j1+j2+j3
#     cond = (int(abs(j1 - j2) <= j3 <= (j1 + j2)) *
#             int(g % 2 == 0))
#     if cond:
#         fronttop = (-1.0)**(0.5*g - j3) * np.sqrt(2*j3+1) * factorial(0.5*g)
#         frontbot = factorial(0.5*g - j1) * factorial(0.5*g - j2) * factorial(0.5*g - j3)
#         back = np.sqrt(factorial(g-2*j1) * factorial(g-2*j2) * factorial(g-2*j3)/factorial(g+1))
#         wanted = (fronttop/frontbot) * back
#         return wanted
#     else:
#         return 0.0


# pylint: disable=too-many-locals
def gettop(ctdict, chrglist, D):
    """
    Creates the top half of the Q matrix.

    Parameters
    ----------
    ctdict : The dictionary corresponding to the T tensor in the original
             HOTRG formulation.
    chrglist : A list of the currect charge values being carried around for
               looping purposes, as well as histogram calculations.
    D : The absolute value of the largest Bessel function used.

    Returns
    -------
    dict : A dictionary parameterizing the tensor which represents the top
           part of the Q matrix.

    """
    small = min(chrglist)
    large = max(chrglist)
    top = {}
    time0 = time()
    for tl, tr in product(chrglist, repeat=2):
        for x in range(-D, D+1):
            _max = max(small, tl + x - D, tr + x - D)
            _min = min(large, tl + x + D, tr + x + D)
            for tp in range(_max, _min + 1):
                t1s = ctdict[tl, x, tp].shape
                t2s = ctdict[tr, x, tp].shape
                ctdict_transpose = np.transpose(
                    ctdict[tl, x, tp], (0, 3, 2, 1)
                )
                temp1 = np.reshape(
                    ctdict_transpose, (t1s[0] * t1s[3], t1s[2] * t1s[1])
                )
                #  temp1 = np.reshape(np.transpose(
                #      ctdict[tl, x, tp], (0, 3, 2, 1)
                #  ), (t1s[0]*t1s[3], t1s[2]*t1s[1]))
                #  temp2 = np.reshape(np.transpose(ctdict[tr, x, tp], (0, 3, 2,
                #                                                      1)),
                #                     (t2s[0]*t2s[3], t2s[2]*t2s[1]))
                _ctdict_transpose = np.transpose(
                    ctdict[tr, x, tp], (0, 3, 2, 1)
                )
                temp2 = np.reshape(
                    _ctdict_transpose, (t2s[0] * t2s[3], t2s[2] * t2s[1])
                )
                try:
                    #  top[(tl, tr+x-tp, tr)] += np.transpose(
                    #      np.reshape(np.dot(temp1, np.transpose(temp2)),
                    #                 (t1s[0], t1s[3], t2s[0], t2s[3])),
                    #      (0, 2, 1, 3)
                    #  )
                    _arr = np.dot(temp1, np.transpose(temp2))
                    _arr = np.reshape(_arr, (t1s[0], t1s[3], t2s[0], t2s[3]))
                    _arr = np.transpose(_arr, (0, 2, 1, 3))
                    top[(tl, tr+x-tp, tr)] += _arr

                except KeyError:
                    #  top[(tl, tr+x-tp, tr)] = np.transpose(
                    #      np.reshape(np.dot(temp1, np.transpose(temp2)),
                    #                 (t1s[0], t1s[3], t2s[0], t2s[3])),
                    #      (0, 2, 1, 3)
                    #  )
                    _arr = np.dot(temp1, np.transpose(temp2))
                    _arr = np.reshape(_arr, (t1s[0], t1s[3], t2s[0], t2s[3]))
                    _arr = np.transpose(_arr, (0, 2, 1, 3))
                    top[(tl, tr+x-tp, tr)] = _arr

    time1 = time()
    print(f"get top time = {time1 - time0}")
    return top

# pylint: disable=too-many-locals
def getbot(ctdict, chrglist, D):
    """
    Creates the bottom half of the Q matrix.

    Parameters
    ----------
    ctdict : The dictionary corresponding to the T tensor in the original
             HOTRG formulation.
    chrglist : A list of the currect charge values being carried around for
               looping purposes, as well as histogram calculations.
    D : The absolute value of the largest Bessel function used.

    Returns
    -------
    dict : A dictionary parameterizing the tensor which represents the bottom
           part of the Q matrix.
           
    """
    bot = {}
    small = min(chrglist)
    large = max(chrglist)
    time0 = time()
    for tl, tr in product(chrglist, repeat=2):
        for xp in range(-D, D+1):
            #  for tp in range(max(small, -D+tl-xp, -D+tr-xp), min(large,
            #  D+tl-xp, D+tr-xp)+1):
            _max = max(small, -D + tl - xp, -D + tr - xp)
            _min = min(large, D + tl - xp, D + tr - xp)
            for tp in range(_max, _min + 1):
                t1s = ctdict[tp, xp, tl].shape
                t2s = ctdict[tp, xp, tr].shape
                #  temp1 = np.reshape(np.transpose(ctdict[tp, xp, tl], (0, 2, 1,
                #                                                       3)),
                #                     (t1s[0]*t1s[2], t1s[1]*t1s[3]))
                _arr = np.transpose(ctdict[tp, xp, tl], (0, 2, 1, 3))
                _arr = np.reshape(_arr, (t1s[0] * t1s[2], t1s[1] * t1s[3]))
                temp1 = _arr
                #  temp2 = np.reshape(np.transpose(ctdict[tp, xp, tr], (0, 2, 1,
                #                                                       3)),
                #                     (t2s[0]*t2s[2], t2s[1]*t2s[3]))
                _arr = np.transpose(ctdict[tp, xp, tr], (0, 2, 1, 3))
                _arr = np.reshape(_arr, (t2s[0] * t2s[2], t2s[1] * t2s[3]))
                temp2 = _arr
                try:
                    #  bot[(tr, tp+xp-tr, tl)] +=
                    #  np.transpose(np.reshape(np.dot(temp1,
                    #  np.transpose(temp2)), (t1s[0], t1s[2], t2s[0], t2s[2])),
                    #               (0, 2, 1, 3))
                    _arr = np.dot(temp1, np.transpose(temp2))
                    _arr = np.reshape(_arr, (t1s[0], t1s[2], t2s[0], t2s[2]))
                    _arr = np.transpose(_arr, (0, 2, 1, 3))
                    bot[(tr, tp + xp - tr, tl)] += _arr
                except KeyError:
                    #  bot[(tr, tp+xp-tr, tl)] =
                    #  np.transpose(np.reshape(np.dot(temp1,
                    #  np.transpose(temp2)), (t1s[0], t1s[2], t2s[0], t2s[2])),
                    #               (0, 2, 1, 3))
                    _arr = np.dot(temp1, np.transpose(temp2))
                    _arr = np.reshape(_arr, (t1s[0], t1s[2], t2s[0], t2s[2]))
                    _arr = np.transpose(_arr, (0, 2, 1, 3))
                    bot[(tr, tp + xp - tr, tl)] = _arr
    time1 = time()
    #  print "get bot time =", (time1-time0)
    print(f'get bot time: {time1 - time0}')
    return bot

# def getQ(top, bot, cvals, D):
#     """
#     Creates a block of the Q matrix provided by charge pair inputs.

#     Parameters
#     ----------
#     top : A dictionary parameterizing the top part of the Q matrix.
#     bot : A dictionary parameterizing the bottom part of the Q matrix.
#     cvals : A list of tuples containing charge pairs that are relavent
#             for building a block of the Q matrix.
#     D : The absolute value of the largest Bessel function used.

#     Returns
#     -------
#     dict : A dictionary with the values of the block of the Q matrix.
#     sizes : A list tuples containing the shape of the two leftmost legs
#             of the Q matrix block.  Used to build the U tensor dictionary.

#     """
#     sizes = []
#     time0 = time()
#     # print cvals
#     for ttr, btr in cvals:
#         temp = []
#         special = False
#         for ttl, btl in cvals:
#             part = 0.0
#             #  for xp in range(max(-D, -D+ttr-ttl, -D+btl-btr), min(D,
#             #  D+ttr-ttl, D+btl-btr)+1):
#             _max = max(-D, -D + ttr - ttl, -D + btl - btr)
#             _min = min(D, D + ttr - ttl, D + btl - btr)
#             for xp in range(_max, _min + 1):
#                 t1s = top[ttl, xp, ttr].shape
#                 t2s = bot[btr, xp, btl].shape
#                 temp1 = np.reshape(
#                     top[ttl, xp, ttr], (t1s[0]*t1s[1], t1s[2]*t1s[3])
#                 )
#                 temp2 = np.reshape(
#                     bot[btr, xp, btl], (t2s[0]*t2s[1], t2s[2]*t2s[3])
#                 )
#                 temp3 = np.dot(temp1, np.transpose(temp2))
#                 _arr = np.reshape(temp3, (t1s[0], t1s[1], t2s[0], t2s[1]))
#                 _arr = np.transpose(_arr, (0, 2, 1, 3))
#                 temp3 = _arr
#                 #  temp3 = np.transpose(np.reshape(temp3, (t1s[0], t1s[1],
#                 #  t2s[0], t2s[1])), (0,2,1,3))
#                 part += np.reshape(temp3, (t1s[0]*t2s[0], t1s[1]*t2s[1]))
#             #  if (((ttl, btl) == cvals[0]) and (type(part) == float)):
#             if (((ttl, btl) == cvals[0]) and (isinstance(part, float))):
#                 special = True
#             elif isinstance(part, float):
#                 pass
#             else:
#                 temp.append(part)
#         if (ttr, btr) == cvals[0]:
#             block = np.vstack(temp)
#             print("cvals[0]", block.shape)
#         else:
#             if special:
#                 temp = np.vstack(temp) # stack the little matricies vertically
#                 #  find out how many rows there are in the current block
#                 ll = (block.shape[0])-(temp.shape[0])
#                 print("special block, temp", block.shape, temp.shape)
#                 print("ll =", ll)
#                 # pads the remaining vertial elements with zeros
#                 temp = np.pad(temp, ((ll, 0), (0, 0)), mode='constant')
#                 print(block.shape, temp.shape)
#                 # combines the new column and the old column together
#                 block = np.hstack((block, temp))
#             else:
#                 temp = np.vstack(temp) # stack the little matricies vertically
#                 # find out how many rows there are in the current block
#                 ll = (temp.shape[0])-(block.shape[0])
#                 print("block, temp", block.shape, temp.shape)
#                 print("ll =", ll)
#                 # pads the remaining vertical elements with zeros
#                 block = np.pad(block, ((0, ll), (0, 0)), mode='constant')
#                 print(block.shape, temp.shape)
#                 # combines the new column and the old column
#                 block = np.hstack((block, temp))
#         sizes.append((t1s[1], t2s[1]))
#         print(block.shape)
#     evals, evecs = np.linalg.eigh(block)
#     time1 = time()
#     return evals, evecs, sizes


def getQ(top, bot, cvals, D):
    """
    Creates a block of the Q matrix provided by charge pair inputs.

    Parameters
    ----------
    top : A dictionary parameterizing the top part of the Q matrix.
    bot : A dictionary parameterizing the bottom part of the Q matrix.
    cvals : A list of tuples containing charge pairs that are relavent
            for building a block of the Q matrix.
    D : The absolute value of the largest Bessel function used.

    Returns
    -------
    dict : A dictionary with the values of the block of the Q matrix.
    sizes : A list tuples containing the shape of the two leftmost legs
            of the Q matrix block.  Used to build the U tensor dictionary.

    """
    sizes = []
    time0 = time()
    # print cvals
    shapeblock = np.zeros((len(cvals), len(cvals), 2))
    squares = {}
    for j, (ttr, btr) in enumerate(cvals):      # columns
        for i, (ttl, btl) in enumerate(cvals):  # rows
            part = 0.0
            for xp in range(max(-D, -D+ttr-ttl, -D+btl-btr), min(D, D+ttr-ttl, D+btl-btr)+1):
                t1s = top[ttl, xp, ttr].shape
                t2s = bot[btr, xp, btl].shape
                temp1 = np.reshape(top[ttl, xp, ttr], (t1s[0]*t1s[1], t1s[2]*t1s[3]))
                temp2 = np.reshape(bot[btr, xp, btl], (t2s[0]*t2s[1], t2s[2]*t2s[3]))
                temp3 = np.dot(temp1, np.transpose(temp2))
                temp3 = np.transpose(np.reshape(temp3, (t1s[0], t1s[1], t2s[0], t2s[1])), (0,2,1,3))
                part += np.reshape(temp3, (t1s[0]*t2s[0], t1s[1]*t2s[1]))
            if (type(part) == float):
                pass
            else:
                squares[(i, j)] = part
                shapeblock[i, :, 0] = part.shape[0]
                shapeblock[:, j, 1] = part.shape[1]
        sizes.append((t1s[1], t2s[1]))
    h = int(np.sum(shapeblock[0, k, 1] for k in range(len(cvals))))
    w = int(np.sum(shapeblock[k, 0, 0] for k in range(len(cvals))))
    block = np.zeros((w, h))
    for key, val in squares.items():
        rr = int(np.sum(shapeblock[k, key[1], 0] for k in range(key[0])))
        cc = int(np.sum(shapeblock[key[0], k, 1] for k in range(key[1])))
        ss = shapeblock[key[0], key[1], :]
        block[rr:rr+int(ss[0]), cc:cc+int(ss[1])] = val
    evals, evecs = np.linalg.eigh(block)
    time1 = time()
    return evals, evecs, sizes


def blockeev(qblock, cvals):
    """
    Calculates the eigenvalues and eigenvectors of a Q matrix block.

    Parameters
    ----------
    qblock : The dictionary containing the values of the block from the
             Q matrix.
    cvals : A list of tuples containing the pairs of charges associated with
            the leftmost legs of the Q matrix block.

    Returns
    -------
    evals : An array of the eigenvalues of the block.
    evecs : An array of the eigenvectors of the block.

    """
    time0 = time()
    for i, j in cvals:
        temp = []
        for k in zip(*cvals):
            temp.append(qblock[i, j, k])
        temp = np.vstack(temp)
        if (i, j) == cvals[0]:
            Qmat = temp
        else:
            Qmat = np.hstack((Qmat, temp))
    a, b = np.linalg.eigh(Qmat)
    time1 = time()
    return a, b

def chrgsets(chrglist):
    """
    Takes a list of charge values and returns all possible charge
    combinations in order, and which indices correspond to those
    charges, also in order.
    
    Parameters
    ----------
    chrglist : A list of charge values, possibly degenerate.
    
    Returns
    -------
    charge_range   : An array of all sums of all possible pairs of charge
                     values one can make with the given list.
    charge_indices : An array of lists whos elements are tuples which
                     gives the pair of charges associated with each
                     tensor leg.

    """
    ls = len(chrglist)
    idxmaster = []
    time0 = time()
    charges = sorted(list(set(
        map(sum, combinations_with_replacement(chrglist, 2))
    )))
    #  charges = sorted(list(set(
    #      imap(sum, combinations_with_replacement(chrglist, 2))
    #  )))
    for charge in charges:
        idxlist = []
        for i, j in product(chrglist, repeat=2):
            if i+j == charge:
                idxlist.append((i, j))
            else:
                pass
        idxmaster.append(idxlist)
    time1 = time()
    print(f"charge build time = {time1 - time0}")#, (time1-time0)
    idxmaster = map(sorted, idxmaster)

    return (charges, idxmaster)

def cass(clist, vlist, idxlist, slist):
    """
    Makes a dictionary associating a charge value to vectors, charge pairs,
    and shapes.

    Parameters
    ----------
    clist : A list of charge values, possibly degenerate.
    vlist : A list of vectors.  Some vectors can share a common charge.
    idxlist : A list of charge-pair tuples. Each charge-pair tuple is
              associated with a tensor shape tuple.
    slist : A list of tuples representing the shape of the leftmost legs of
            a block from the Q matrix.  Each shape tuple is associated with
            a charge pair.

    Returns
    -------
    dict : A dictionary containing all of the vectors, charge-pairs, and
          shape-pairs associated with a charge value (block).

    """
    vecs = {}
    time0 = time()
    # pylint:disable=invalid-name
    for c, v, i, s in zip(clist, vlist, idxlist, slist):
        try:
            vecs[c][0] = np.vstack((vecs[c][0], v))
            # temp = vecs[c][0].shape
            # vecs[c][0] = np.reshape(vecs[c][0], temp)
        except KeyError:
            vecs[c] = [v, i, s]
            temp = len(vecs[c][0])
            vecs[c][0] = np.reshape(vecs[c][0], (-1, temp))
    time1 = time()
    print(f"get cass time = {time1 - time0}")#, (time1-time0)
    return vecs

def getU(vecs):
    """
    Creates a dictionary containing the non-zero elements of the U tensor
    used for updating.

    Parameters
    ----------
    vecs : A dictionary which associates the necessary vectors, charge
           charge-pairs, and shape-pairs with a charge used to build the U
           tensor.

    Returns
    -------
    dict : A dictionary with the non-zero elements of the U tensor used for
           updating.  It is parameterized with two charge values which fix the
           output leg.

    Notes
    -----
    If this is correct the output order for the sub-indices is
    (top, bot, alpha).

    """
    cvdict = {}
    time0 = time()
    for c in vecs:
        s = 0
        e = 0
        for i, j in zip(vecs[c][1], vecs[c][2]):
            e += prod(j)
            xx = (vecs[c][0][:, s:e]).shape
            # xx = np.transpose(vecs[c][0][:,s:e]).shape
            #print xx
            #print (j + (xx[1],))
            # cvdict[i] = np.reshape(np.transpose(vecs[c][0][:,s:e]), (j +
            # (xx[1],)))
            cvdict[i] = np.reshape(
                np.transpose(vecs[c][0][:, s:e]), (j + (xx[0],))
            )
            s = e
    time1 = time()
    print(f"make U time = {time1 - time0}")#, (time1-time0)
    return cvdict

def update(ttop, tbot, udict, cidladded, D):
    """
    Creates a dictionary with the non-zero elements of an updated tensor after
    having gone through an iteration of HOTRG.

    Parameters
    ----------
    ctdict : A dictionary parameterizing the T tensor.
    udict : A dictionary parameterizing the U tensor.
    cidladded : A list of charge-pairs used for updating.  These are the
                reduced selected charge pairs which are used by udict.
    D : The absolute value of the largest Bessel function used.

    Returns
    -------
    dict : A dictionary parameterizing the updated T tensor.  It is
           parameterized using three charge values again which imply the
           fourth.
    """
    topret = {}
    botret = {}
    time0 = time()
    for tl, bl in cidladded:
        for tr, br in cidladded:
            _max = max(-D, -D + tl - tr, -D + br - bl)
            _min = min(D, D + tl - tr, D + br - bl)
            for k in range(_max, _min + 1):
                uls = udict[tl, bl].shape
                tts = tbot[tr, k, tl].shape
                ###############################################################
                # NOTE: IF IT'S NOT WORKING, THIS SECTION MIGHT BE THE ISSUE 
                ###############################################################
                arr1 = np.reshape(
                    np.transpose(udict[tl, bl], (2, 1, 0)),
                    (uls[2] * uls[1], uls[0])
                )
                arr2 = np.reshape(
                    tbot[tr, k, tl],
                    (tts[0], tts[1] * tts[2] * tts[3])
                )
                top = np.dot(arr1, arr2)


                urs = udict[tr, br].shape
                tbs = ttop[bl, k, br].shape

                arr1 = np.reshape(
                    np.transpose(ttop[bl, k, br], (0, 3, 2, 1)),
                    (tbs[0] * tbs[3] * tbs[2], tbs[1])
                )
                arr2 = np.reshape(
                    np.transpose(udict[tr, br], (1, 0, 2)),
                    (urs[1], urs[0] * urs[2])
                )
                bot = np.dot(arr1, arr2)

                arr1 = np.reshape(
                    bot, (tbs[0], tbs[3], tbs[2], urs[0], urs[2])
                )
                arr1 = np.transpose(arr1, (3, 0, 2, 1, 4))

                bot = np.reshape(
                    arr1, (urs[0] * tbs[0] * tbs[2], tbs[3] * urs[2])
                )

                arr2 = np.reshape(
                    top, (uls[2], uls[1], tts[1], tts[2], tts[3])
                )
                arr2 = np.transpose(arr2, (0, 3, 2, 1, 4))
                top = np.reshape(
                    arr2, (uls[2] * tts[2],  tts[1] * uls[1] * tts[3])
                )
                
                topret_idx = (tl + bl, k + tr - tl, tr + br)
                dot_prod = np.dot(top, bot)
                
                arr1 = np.reshape(
                    dot_prod,
                    (uls[2], tts[2], tbs[3], urs[2])
                    )
                arr1 = np.transpose(arr1, (0, 3, 1, 2))
                try:
#                     topret_idx = (tl + bl, k + tr - tl, tr + br)
#                     dot_prod = np.dot(top, bot)

#                     arr1 = np.reshape(
#                         dot_prod,
#                         (uls[2], tts[2], tbs[3], urs[2])
#                     )
#                     arr1 = np.transpose(arr1, (0, 3, 1, 2))
                    topret[topret_idx] += arr1

#                     botret_idx = (tr + br, bl + k - br, tl + bl)
#                     botret[botret_idx] = topret[topret_idx]
                except KeyError:
#                     topret_idx = (tl + bl, k + tr - tl, tr + br)
#                     dot_prod = np.dot(top, bot)

#                     arr1 = np.reshape(
#                         dot_prod,
#                         (uls[2], tts[2], tbs[3], urs[2])
#                     )
#                     arr1 = np.transpose(arr1, (0, 3, 1, 2))
                    topret[topret_idx] = arr1

                botret_idx = (tr + br, bl + k - br, tl + bl)
                    #  topret_idx = (tl + bl, k + tr - tl, tr + br)
                botret[botret_idx] = topret[topret_idx]
    time1 = time()
    print(f"update tensor time = {time1 - time0}")#, (time1-time0)
    return topret, botret

def tensor_norm(ctdict, chrglist, D):
    """Normalize tensor."""
    norm = 0.0
    for i, j in product(chrglist, repeat=2):
        _max = max(-D, -D + j - i)
        _min = min(D, D + j - i)
        for k in range(_max, _min + 1):
            norm += np.einsum('abcd, abcd', ctdict[i, k, j], ctdict[i, k, j])
    norm = np.sqrt(norm)
    return norm

def normalize(d, norm):
    """Normalize `d`."""
    for k, v in d.items():
        d[k] = v/norm

def tensor_trace(ctdict, chrglist, D):
    """Trace of tensor."""
    trace = 0.0
    for i in chrglist:
        for k in range(-D, D+1):
            trace += np.einsum('aa', np.einsum('aaij', ctdict[i, k, i]))
    return trace

def dict_transpose(thedict, chrglist, D):
    """Transpose dictionary."""
    newdict = {}
    for tl, tr in product(chrglist, repeat=2):
        for x in range(max(-D, -D+tr-tl), min(D, D+tr-tl)+1):
            newdict[(tr, tl+x-tr, x)] = thedict[(tl, x, tr)]
    return newdict

# pylint:disable=invalid-name, too-many-locals, too-many-arguments
def getlists(O, Top, Bot, D, rdbond, dbond):
    """
    Returns the good values of charge, vectors, charge pairs, and pair shapes. 

    Parameters
    ----------
    O : A tuple of lists of charges and charge pairs.
    Top : The top part of the Q matrix.
    Bot : The bottom part of the Q matrix.
    D : The absolute value of the largest Bessel used.

    Returns
    -------
    clist : A list of the charges that made it through the slection.
    vlist : A list of vectors that made is through the selection.
    idxlist : The pairs of charges that match with the vectors and charges.
    slist : A list of the sizes of the charge pairs.
    """
    rdbond = min(rdbond**2, dbond)
    print(f"local rdbond = {rdbond}")#, rdbond
    clist = list(np.zeros((rdbond)))  # initialize lists to be filled
    elist = list(np.zeros((rdbond)))
    vlist = list(np.zeros((rdbond)))
    idxlist = list(np.full((rdbond), 0.1))
    slist = list(np.zeros((rdbond)))
    globaltime0 = time()
    for c, l in O:  # loop through charges and pairs
        e, v, sizes = getQ(Top, Bot, l, D)  # make Q and the pair sizes
        idx = e.argsort()
        e = e[idx]
        v = v[:, idx].T
        #  for i in range(len(e)):
        for idx, ei in enumerate(e):
            #  if e[i] >= min(elist):  # fill the lists with the best e and v
            if ei >= min(elist):
                j = np.argmin(elist)
                elist.pop(j)
                #  elist.append(e[i])
                elist.append(ei)
                idxlist.pop(j)
                idxlist.append(l)
                clist.pop(j)
                clist.append(c)
                vlist.pop(j)
                #  vlist.append(v[i])
                vlist.append(v[idx])
                slist.pop(j)
                slist.append(sizes)
            else:
                pass
    globaltime1 = time()
    print(f"total Q and eigs time = {globaltime1 - globaltime0}")
    #, (globaltime1-globaltime0)
    return clist, vlist, idxlist, slist

#pylint: disable=invalid-name,too-many-locals
# def maketm(ttop, tbot, chrglist, D):
#     """
#     #####################################################
#     # NOTE: This funtion is not ready yet, empty blocks #
#     #        need an exception.                         #
#     ####################################################
#     Takes the current tensor and contracts it with itself to make a transfer
#     matrix.

#     Parameters
#     ----------
#     ttop :  The tensor.  This is the tensor which is parameterized in terms
#             of its left, right, and top inidices.
#     tbot :  The tensor.  This is the tensor which is parameterized in terms of
#             its left, right, and bottom inidices.
#     chrglist : This is a list of lists of charge pair tuples.  The lists are
#         sorted by total charge of the tuple pairs.
#     D :        The abolute value of the largest Bessel used in the begining.

#     Returns
#     -------
#     TM : The transfer matrix in block diagonal form in terms of charge sectors.

#     """
#     blocks = []
#     for cid in chrglist:
#         # print cid
#         for ttl, btl in cid:
#             # print "----new row----"
#             temp = []
#             for ttr, btr in cid:
#                 # print "new col."
#                 part = 0.0
#                 _max = max(-D, -D + ttl - ttr, -D + btr - btl)
#                 _min = min(D, D + ttl - ttr, D + btr - btl)
#                 for xp in range(_max, _min + 1):
#                     ts = tbot[ttr, xp, ttl].shape
#                     bs = ttop[btl, xp, btr].shape
#                     top = np.reshape(
#                         tbot[ttr, xp, ttl],
#                         (ts[0]*ts[1], ts[2]*ts[3])
#                     )
#                     bot = np.reshape(
#                         np.transpose(ttop[btl, xp, btr], (0, 1, 3, 2)),
#                         (bs[0]*bs[1], bs[3]*bs[2])
#                     )
#                     part += np.reshape(
#                         np.transpose(np.reshape(np.dot(top, bot.T),
#                                                 (ts[0], ts[1], bs[0], bs[1])),
#                                      (0, 2, 1, 3)), (ts[0]*bs[0], ts[1]*bs[1])
#                     )
#                 if isinstance(part, float):
#                     pass
#                 else:
#                     temp.append(part)
#             if (ttl, btl) == cid[0]:
#                 print([len(x) for x in temp])
#                 tm = np.hstack(temp)
#                 print(tm.shape)
#             else:
#                 tm = np.vstack((tm, np.hstack(temp)))
#         blocks.append(tm)
#     return block_diag(*np.array(blocks))


def make_tm(ttop, tbot, chrglist, D):
    """
    Takes the current tensor and contracts it with itself to make the
    eigenvalues of the transfer matrix, the density matrix, and the reduced
    density matrix.

    Parameters
    ----------
    ttop :     The tensor.  This is the tensor which is parameterized in terms
               of its left, right, and top inidices.
    tbot :     The tensor.  This is the tensor which is parameterized in terms
               of its left, right, and bottom inidices.
    chrglist : This is a list of lists of charge pair tuples.  The lists are
               sorted by total charge of the tuple pairs.
    D :        The abolute value of the largest Bessel used in the begining.
    #  titer :    The number of iterations to take in the time direction, Nt =
    #  2**titer.

    Returns
    -------
    chrg_evals : The eigenvalues of the charge blocks of the transfer matrix
                 and their charge as a dict.
    rdmchrg :    The eigenvalues of the reduced density matrix and their
                 charge.
    dmchrg :     The eigenvalues of the density matrix and their charge.
    """
    chrg_evals = {}
#     rdmchrg = {}
#     dmchrg = {}

    # this will store the shapes of the diagonal blocks by charge 
    sblock = {}
    # this will store the tensor shapes of the diagonal blocks by charge
    tsblock = {}
    blocks = {}
    for cid in chrglist:
        # print cid
        # the current shape storing array
        shapeblock = np.zeros((len(cid), len(cid), 2))
        # the current tensor shape storing array
        tshapeblock = np.zeros((len(cid), len(cid), 4))
        squares = {}
        for i, (ttl, btl) in enumerate(cid):
            # print "----new row----"
            for j, (ttr, btr) in enumerate(cid):
                # print "new col."
                part = 0.0
                _max = max(-D, -D + ttl - ttr, -D + btr - btl)
                _min = min(D, D + ttl - ttr, D + btr - btl)
                for xp in range(_max, _min + 1):
#                     if ((ttr + xp - ttl) == 0):
                        # this is the tensor parameterized by the bottom idxs
                    ts = tbot[ttr, xp, ttl].shape
                    #  this is the tensor parameterized by the top idxs
                    bs = ttop[btl, xp, btr].shape
                    top = np.reshape(
                        tbot[ttr, xp, ttl],
                        (ts[0]*ts[1], ts[2]*ts[3])
                        )
                    bot = np.reshape(
                        np.transpose(ttop[btl, xp, btr], (0, 1, 3, 2)),
                        (bs[0]*bs[1], bs[3]*bs[2])
                        )
                    part += np.reshape(
                        np.transpose(np.reshape(np.dot(top, bot.T),
                                                (ts[0], ts[1], bs[0], bs[1])),
                                     (0, 2, 1, 3)), (ts[0]*bs[0], ts[1]*bs[1])
                        )
#                 else:
#                         pass
                if isinstance(part, float):
                    pass
                else:
                    squares[(i, j)] = part
                    shapeblock[i, :, 0] = part.shape[0]
                    shapeblock[:, j, 1] = part.shape[1]
                    tshapeblock[i, j, :] = np.array([
                        ts[0], bs[0], ts[1], bs[1]
                    ])
        h = int(np.sum(shapeblock[0, k, 1] for k in range(len(cid))))
        w = int(np.sum(shapeblock[k, 0, 0] for k in range(len(cid))))
        # print "width and height of block =", w, h
        tm = np.zeros((w, h))
        for key, val in squares.items():
            # print "key =", key
            rr = int(np.sum(shapeblock[k, key[1], 0] for k in range(key[0])))
            # print "to row ", rr
            cc = int(np.sum(shapeblock[key[0], k, 1] for k in range(key[1])))
            # print "to col ", cc
            ss = shapeblock[key[0], key[1], :]
            # print "shape =", ss
            # print "final two =", rr+ss[0], cc+ss[1]
            tm[rr:rr+int(ss[0]), cc:cc+int(ss[1])] = val
        sblock[np.sum(cid[0])] = shapeblock
        tsblock[np.sum(cid[0])] = tshapeblock
        # print "tm shape =", tm.shape
        # compute the eigvals of the transfer matrix block
        chrg_evals[np.sum(cid[0])] = np.sort(np.linalg.eigvalsh(tm))[::-1]
        blocks[np.sum(cid[0])] = tm
    return chrg_evals, blocks
