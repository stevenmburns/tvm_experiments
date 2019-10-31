import numpy as np
import numpy.linalg as la

import functools
from itertools import product, chain


class Stats:
    def __init__(self):
        self.a_loads = 0
        self.b_loads = 0
        self.c_loads = 0
        self.c_stores = 0

        self.a_storage = 0
        self.b_storage = 0
        self.c_storage = 0

    def compute_storage(self, tag, a):
        setattr( self, tag + '_storage', functools.reduce(lambda a, b: a*b, a.shape))

    @property
    def loads(self):
        return self.a_loads + self.b_loads + self.c_loads

    @property
    def total_storage(self):
        return self.a_storage + self.b_storage + self.c_storage

    def prnt(self):

        print(f"a,b,c_loads: {self.a_loads},{self.b_loads},{self.c_loads} c_stores: {self.c_stores} a,b,c_storage: {self.a_storage},{self.b_storage},{self.c_storage} loads: {self.loads} total_storage: {self.total_storage}")


def test_A():

    N,M,L = 64,64,64

#    bn,bm,bl = 16,16,16
    bn,bm,bl = 32,1,16

    assert N % bn == 0
    assert M % bm == 0
    assert L % bl == 0

    def mm0(A, B):
        return np.dot(A, B)

    def mm1(A, B):
        s = Stats()
        s.a_storage, s.b_storage, s.c_storage = 1, 1, 1

        C = np.zeros(shape=(N, M))
        for i in range(N):
            for j in range(M):
                for k in range(L):
                    C[i, j] += A[i, k]*B[k, j]
                    s.a_loads += 1
                    s.b_loads += 1
                    s.c_loads += 1
                    s.c_stores += 1

        s.prnt()

        return C

    def mm2(A, B):
        """N*L*M loads and stores"""
        s = Stats()
        s.a_storage, s.b_storage, s.c_storage = 1, 1, 1
        C = np.zeros(shape=(N, M))
        for I in range(N//bn):
            for J in range(M//bm):
                for K in range(L//bl):
                    for i in range(bn):
                        for j in range(bm):
                            for k in range(bl):
                                ii = bn*I + i
                                jj = bm*J + j
                                kk = bl*K + k
                                C[ii, jj] += A[ii, kk]*B[kk, jj]
                                s.a_loads += 1
                                s.b_loads += 1
                                s.c_loads += 1
                                s.c_stores += 1

        s.prnt()

        return C

    def mm3(A, B):
        """
a_loads: N//bn * L//bl * bn * bl = N * L
b_loads: N//bn * L//bl * M//bm * bl * bm = N//bn * L * M
c_stores: N//bn * L//bl * M//bm * bn * bm = N * L//bl * M
n
Ratios: M, bn, bl
"""
        s = Stats()

        C = np.zeros(shape=(N, M))
        for I in range(N//bn):
            for K in range(L//bl):
                AA = np.zeros(shape=(bn, bl))
                s.compute_storage('a', AA)
                for i in range(bn):
                    for k in range(bl):
                        AA[i, k] = A[bn*I+i, bl*K+k]
                        s.a_loads += 1
                for J in range(M//bm):
                    BB = np.zeros(shape=(bl, bm))
                    s.compute_storage('b', BB)
                    for k in range(bl):
                        for j in range(bm):
                            BB[k, j] = B[bl*K+k, bm*J+j]
                            s.b_loads += 1
                    CC = np.zeros(shape=(bn, bm))
                    s.compute_storage('c', CC)
                    for i in range(bn):
                        for j in range(bm):
                            CC[i, j] = C[bn*I + i, bm*J + j]
                            s.c_loads += 1
                    for k in range(bl):
                        for i in range(bn):
                            for j in range(bm):
                                CC[i, j] += AA[i, k]*BB[k, j]
                    for i in range(bn):
                        for j in range(bm):
                            C[bn*I + i, bm*J + j] = CC[i, j]
                            s.c_stores += 1

        s.prnt()
        return C

    def mm_share_a(A, B):
        """
a_loads: N//bn * L//bl * bn * bl = N * L
b_loads: N//bn * L//bl * M//bm * bl * bm = N//bn * L * M
c_stores: N//bn * L//bl * M//bm * bn * bm = N * L//bl * M
Ratios: M, bn, bl
"""
        s = Stats()

        C = np.zeros(shape=(N, M))
        for K, I in product(range(L//bl), range(N//bn)):
            AA = np.zeros(shape=(bn, bl))
            s.compute_storage('a', AA)
            for i, k in product(range(bn), range(bl)):
                AA[i, k] = A[bn*I+i, bl*K+k]
                s.a_loads += 1
            for J in range(M//bm):
                BB = np.zeros(shape=(bl, bm))
                s.compute_storage('b', BB)
                for k, j in product(range(bl), range(bm)):
                    BB[k, j] = B[bl*K+k, bm*J+j]
                    s.b_loads += 1
                CC = np.zeros(shape=(bn, bm))
                s.compute_storage('c', CC)
                for i, j in product(range(bn), range(bm)):
                    CC[i, j] = C[bn*I + i, bm*J + j]
                    s.c_loads += 1
                for k, i, j in product(range(bl), range(bn), range(bm)):
                    CC[i, j] += AA[i, k]*BB[k, j]
                for i, j in product(range(bn), range(bm)):
                    C[bn*I + i, bm*J + j] = CC[i, j]
                    s.c_stores += 1

        s.prnt()
        return C

    def mm_share_c(A, B):
        """
a_loads: N * M//bm * L
b_loads: N//bn * M * L
c_loads: N * M
c_stores: N * M
Ratios: M, bn, bl
"""
        s = Stats()

        C = np.zeros(shape=(N, M))

        for I in range(N//bn):
            for J in range(M//bm):
              CC = np.zeros(shape=(bn, bm))
              s.compute_storage('c', CC)
              for i, j in product(range(bn), range(bm)):
                  CC[i, j] = C[bn*I + i, bm*J + j]
                  s.c_loads += 1
              for K in range(L//bl):
                  AA = np.zeros(shape=(bn, bl))
                  s.compute_storage('a', AA)
                  for i, k in product(range(bn), range(bl)):
                      AA[i, k] = A[bn*I+i, bl*K+k]
                      s.a_loads += 1
                  BB = np.zeros(shape=(bl, bm))
                  s.compute_storage('b', BB)
                  for k, j in product(range(bl), range(bm)):
                      BB[k, j] = B[bl*K+k, bm*J+j]
                      s.b_loads += 1
                  for k, i, j in product(range(bl), range(bn), range(bm)):
                      CC[i, j] += AA[i, k]*BB[k, j]
              for i, j in product(range(bn), range(bm)):
                  C[bn*I + i, bm*J + j] = CC[i, j]
                  s.c_stores += 1

        s.prnt()
        return C

    A = np.random.uniform(size=(N, L))
    B = np.random.uniform(size=(L, M))

#    print('mm0-mm1')
#    assert la.norm(mm0(A, B)-mm1(A, B)) < 1e-5
#    print('mm0-mm2')
#    assert la.norm(mm0(A, B)-mm2(A, B)) < 1e-5
#    print('mm0-mm3')
#    assert la.norm(mm0(A, B)-mm3(A, B)) < 1e-5
    print('mm0-mm_share_a')
    assert la.norm(mm0(A, B)-mm_share_a(A, B)) < 1e-5
    print('mm0-mm_share_c')
    assert la.norm(mm0(A, B)-mm_share_c(A, B)) < 1e-5
