import argparse

import o3funcs as csf
from time import time
from pprint import pprint
from sys import argv

import numpy as np

# pylint:disable=too-many-statements,invalid-name
def main(args):
    """`main` method for running module as script."""
    #  lmax = int(argv[1])    # the largest total angular momentum value
    #  D = lmax    # the absolute value of the largest Bessel function used
    #  rDbond = (lmax+1)**2    # the initial/running Dbond
    #  Dbond = int(argv[2])    # the ultimate Dbond
    #  beta = np.float64(argv[3])
    #  mu = np.float64(argv[4])
    #  nit = int(argv[5])    # The number of iterations

    lmax = args.lmax  # the largest total angular momentum value
    D = lmax          # absolute val of largest Bessel function used
    rDbond = (lmax + 1) ** 2    # the intial/running Dbond
    Dbond = args.Dbond          # the ultimate Dbond
    beta = args.beta
    mu = args.mu
    nit = args.nit

    print("------------------------------------")
    print(f"lmax = {lmax}")
    print(f"D = {D}")
    print(f"initial Dbond = {rDbond}")
    print(f"Dbond = {Dbond}")
    print(f"Beta = {beta}")
    print(f"Mu = {mu}")
    print(f"iterations = {nit}")
    print("------------------------------------")

    runchrglist = []  # keeps the running values of histogram charges
    initchrg = range(-D, D+1)       # the initial charges
    runchrglist.append(initchrg)
    totalgtime0 = time()            # begin total timing
    T, B = csf.tensorgen(beta, mu, D, lmax)   # generate the initial tensors
    print(f"partition function = {csf.tensor_trace(T, initchrg, D)}")
    norm = csf.tensor_norm(T, initchrg, D)
    print(f"normalization= {norm}")
    csf.normalize(T, norm)              # normalize the tensors
    csf.normalize(B, norm)
    # timelist = []   # This list stores the largest looping times

    for i in range(nit):
        if i == nit-1:
            print("------------------------------------")
            print(f"iteration {i+1}")
            print("------------------------------------")
            charges, cidl = csf.chrgsets(initchrg) # generate charges and pairs
            evals, tm_blocks = csf.make_tm(T, B, cidl, D)
        else:
            print("------------------------------------")
            print(f"iteration {i+1}")
            print("------------------------------------")
            print(f"current Dbond = {rDbond}")
            charges, cidl = csf.chrgsets(initchrg) # generate charges and pairs
            O = zip(charges, cidl)
            Top = csf.gettop(T, initchrg, D) # make the top of the Q matrix
            Bot = csf.getbot(B, initchrg, D) # make the bottom of the Q matrix
            # makes all the lists I need
            clist, vlist, idxlist, slist = csf.getlists(O, Top, Bot,
                                                        D, rDbond, Dbond)
            rDbond = len(clist)

            # puts all the lists together so they are associated
            tempcass = csf.cass(clist, vlist, idxlist, slist)

            U = csf.getU(tempcass)  # gets the U for updating the tensor
            cidladded = csf.allcidl(idxlist)
            lt1 = time()
            T, B = csf.update(T, B, U, cidladded, D) # update original tensors
            lt2 = time()
            # timelist.append(lt2-lt1)
            initchrg = list(set(clist))
            runchrglist.append(clist)   # appends the current histogram charges
            norm = csf.tensor_norm(T, initchrg, D)
            print(f"normalization = {norm}")
            print(f"partition function = {csf.tensor_trace(T, initchrg, D)}")
            csf.normalize(T, norm)
            csf.normalize(B, norm)

    
    print(evals)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('')
    )
    parser.add_argument("--lmax", type=int, dest="lmax",
                        help="Largest total angular momentum value.")

    parser.add_argument("--Dbond", type=int, dest="Dbond",
                        help="The ultimate Dbond.")

    parser.add_argument("--beta", type=np.float64, dest="beta",
                        help="beta")

    parser.add_argument("--mu", type=np.float64, dest="mu",
                        help="mu")

    parser.add_argument("--nit", type=int, dest="nit",
                        help="Number of iterations.")

    args = parser.parse_args()

    main(args)
