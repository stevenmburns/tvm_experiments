import matplotlib.pyplot as plt
import numpy as np
import math

from gpkit import Variable, Model

N,M,L = 64,64,64

def f_a( N, M, L, bn, bm, bl):
    return N*L + N*M*L/bn + 2*N*M*L/bl

def f_c( N, M, L, bn, bm, bl):
    return N*M*L/bn + N*M*L/bm + N*M

def g( N, M, L, bn, bm, bl):
    return bn*bl + bl*bm + bn*bm

def optimal_value( f, g):
    cN = Variable('N', N)
    cM = Variable('M', M)
    cL = Variable('L', L)
    S = Variable( 'S', 3.0*256)

    bn = Variable( 'bn')
    bm = Variable( 'bm')
    bl = Variable ( 'bl')

    objective = f( cN, cM, cL, bn, bm, bl)
    storage = g( cN, cM, cL, bn, bm, bl)
    
    m = Model( objective, [storage <= S, bm >=1, bn >=1, bl >=1])

    sol = m.solve(verbosity=1)

    return sol(bn), sol(bm), sol(bl)


xs = np.linspace( 0, 6, 21)

bs = [ 2**i for i in xs]

z_a = [ [ f_a(N,M,L,bn,1,bl) for bn in bs] for bl in bs]
z_c = [ [ f_c(N,M,L,bn,bm,1) for bn in bs] for bm in bs]
z_g   = [ [ g(N,M,L,bn,bm,1) for bn in bs] for bm in bs]

m_a = min( (min(row) for row in z_a))
m_c = min( (min(row) for row in z_c))

q = [1]
for i in range(40):
   q.append( q[-1]*1.1)

plt.clf()
o = optimal_value( f_a, g)
plt.plot( [math.log2(o[0])], [math.log2(o[2])], 'o')
plt.contour( xs, xs, z_a, levels=[m_a*x for x in q], colors='red')
plt.contour( xs, xs, z_g, levels=[256,2*256,3*256], colors='blue')
plt.title(f"f_a {m_a}")
plt.xlabel("bn")
plt.ylabel("bl")
plt.axes().set_aspect('equal')
plt.savefig( 'f_a.png')

plt.clf()
o = optimal_value( f_c, g)
plt.plot( [math.log2(o[0])], [math.log2(o[1])], 'o')
plt.contour( xs, xs, z_c, levels=[m_c*x for x in q], colors='green')
plt.contour( xs, xs, z_g, levels=[256,2*256,3*256], colors='blue')
plt.title(f"f_c {m_c}")
plt.xlabel("bn")
plt.ylabel("bm")
plt.axes().set_aspect('equal')
plt.savefig( 'f_c.png')

#plt.show()


