import numpy as np
import numpy.linalg as la




def test_A():

    s = 1
    t = 4

    N = 64*s
    M = 64*s
    L = 64*s

    bn = 8*t
    bm = 8*t
    bl = 8*t

    assert N%bn == 0
    assert M%bm == 0
    assert L%bl == 0

    def mm0( A, B):
        return np.dot( A, B)

    def mm1( A, B):
        C = np.zeros( shape=(N,M))
        for i in range(N):
            for j in range(M):
                for k in range(L):
                    C[i,j] += A[i,k]*B[k,j]
        return C

    def mm2( A, B):
        """N*L*M loads and stores"""
        a_loads,b_loads,c_loads,c_stores = 0,0,0,0
        a_storage,b_storage,c_storage = 1,1,1
        C = np.zeros( shape=(N,M))
        for I in range(N//bn):
            for J in range(M//bm):
                for K in range(L//bl):
                    for i in range(bn):
                        for j in range(bm):
                            for k in range(bl):
                                ii = bn*I + i
                                jj = bm*J + j
                                kk = bl*K + k
                                C[ii,jj] += A[ii,kk]*B[kk,jj]
                                a_loads += 1
                                b_loads += 1
                                c_loads += 1
                                c_stores += 1
                                
        print( f"a,b,c_loads: {a_loads},{b_loads},{c_loads} c_stores: {c_stores} a,b,c_storage: {a_storage},{b_storage},{c_storage}")

        return C

    def mm3( A, B):
        """
a_loads: N//bn * L//bl * bn * bl = N * L
b_loads: N//bn * L//bl * M//bm * bl * bm = N//bn * L * M
c_stores: N//bn * L//bl * M//bm * bn * bm = N * L//bl * M

Ratios: M, bn, bl
"""
        a_loads,b_loads,c_loads,c_stores = 0,0,0,0

        C = np.zeros( shape=(N,M))
        for I in range(N//bn):
            for K in range(L//bl):
                AA = np.zeros( shape=(bn,bl))
                a_storage = AA.shape[0] * AA.shape[1]
                for i in range(bn):
                    for k in range(bl):
                        AA[i,k] = A[bn*I+i,bl*K+k]
                        a_loads += 1
                for J in range(M//bm):
                    BB = np.zeros( shape=(bl,bm))
                    b_storage = BB.shape[0] * BB.shape[1]
                    for k in range(bl):
                        for j in range(bm):
                            BB[k,j] = B[bl*K+k,bm*J+j]
                            b_loads += 1
                    CC = np.zeros( shape=(bl,bm))
                    c_storage = CC.shape[0] * CC.shape[1]
                    for i in range(bn):
                        for j in range(bm):
                            CC[i,j] = C[bn*I + i,bm*J + j]
                            c_loads += 1
                    for k in range(bl):
                        for i in range(bn):
                            for j in range(bm):
                                CC[i,j] += AA[i,k]*BB[k,j]
                    for i in range(bn):
                        for j in range(bm):
                            C[bn*I + i,bm*J + j] = CC[i,j]
                            c_stores += 1
                    
        print( f"a,b,c_loads: {a_loads},{b_loads},{c_loads} c_stores: {c_stores} a,b,c_storage: {a_storage},{b_storage},{c_storage}")
        return C



    A = np.random.uniform( size=(N,L))
    B = np.random.uniform( size=(L,M))
    
    print('mm0-mm1')
    assert la.norm( mm0(A,B)-mm1(A,B)) < 1e-5
    print('mm0-mm2')
    assert la.norm( mm0(A,B)-mm2(A,B)) < 1e-5
    print('mm0-mm3')
    assert la.norm( mm0(A,B)-mm3(A,B)) < 1e-5


    
