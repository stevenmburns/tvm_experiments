
import pytest

from gpkit import Variable, Model

@pytest.fixture
def fixture():
    # make the constant a float (important)
    N = Variable( 'N', 64.0)
    M = Variable( 'M', 64.0)
    L = Variable( 'L', 64.0)
    S = Variable( 'S', 3.0*256)

    bn = Variable( 'bn')
    bm = Variable( 'bm')
    bl = Variable ( 'bl')

    return N,M,L,S,bn,bm,bl

def test_A(fixture):
    N,M,L,S,bn,bm,bl = fixture

    share_a = N*L + N*M*L/bn + 2*N*M*L/bl
    storage = bn*bl + bl*bm + bn*bm

    m = Model( share_a, [storage <= S, bm >=1, bn >=1, bl >=1])

    sol = m.solve(verbosity=1)

    print("share_a",sol.table())

def test_C(fixture):
    N,M,L,S,bn,bm,bl = fixture

    share_c = N*M*L/bn + N*M*L/bm + N*M
    storage = bn*bl + bl*bm + bn*bm

    m = Model( share_c, [storage <= S, bm >=1, bn >=1, bl >=1])

    sol = m.solve(verbosity=1)

    print('share_c', sol.table())
