from __future__ import absolute_import, print_function

import tvm
import numpy as np
from IPython import embed
from topi.util import get_const_tuple


def testit():
    #
    # Make sure it is the same as np.dot( a.T, b)
    #
    target = "llvm -mcpu=core-avx2"
    func = tvm.build(s, [A, B, C], target=target, name="outer_product")
    a = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
    b = np.random.uniform(size=get_const_tuple(B.shape)).astype(A.dtype)
    ctx = tvm.context(target, 0)
    c = tvm.nd.array(np.zeros(get_const_tuple(C.shape), dtype=A.dtype), ctx)
    func(tvm.nd.array(a, ctx), tvm.nd.array(b, ctx), c)
    tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.T, b), rtol=1e-3)


N, M, L = 1024, 512, 64

if False:
    #
    # The following lines describe the computation :code:`A^T * B` in TVM.
    #

    # first argument is a 2D shape
    A = tvm.placeholder((L, N), name='A')
    B = tvm.placeholder((L, M), name='B')
    # first argument is a 1D range
    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j:
                    tvm.sum(A[k, i] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    testit()

    #
    # Make it outer_product by moving z above xi,yi
    #
    factor = 16
    x, y = C.op.axis
    z, = C.op.reduce_axis
    xo, xi = s[C].split(x, factor=factor)
    yo, yi = s[C].split(y, factor=factor)
    s[C].reorder(xo, yo, z, xi, yi)
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    testit()

# first argument is a 2D shape
A = tvm.placeholder((L, N), name='A')
B = tvm.placeholder((L, M), name='B')

A_buf = tvm.compute((L, N), lambda *i: A(*i), name='A_buf')
B_buf = tvm.compute((L, M), lambda *i: B(*i), name='B_buf')

# first argument is a 1D range
k = tvm.reduce_axis((0, L), name='k')
C_buf = tvm.compute((N, M), lambda i, j:
                    tvm.sum(A_buf[k, i] * B_buf[k, j], axis=k), name='C_buf')
C = tvm.compute((N, M), lambda *i: C_buf(*i), name='C')
s = tvm.create_schedule(C.op)

print(tvm.lower(s, [A, B, C], simple_mode=True))
testit()

factor = 16
x, y = C_buf.op.axis
z, = C_buf.op.reduce_axis
xo, xi = s[C_buf].split(x, factor=factor)
yo, yi = s[C_buf].split(y, factor=factor)
s[C_buf].reorder(z, xo, yo, xi, yi)

print(tvm.lower(s, [A, B, C], simple_mode=True))
testit()


s[A_buf].compute_at(s[C_buf], yo)
s[B_buf].compute_at(s[C_buf], yo)
print(tvm.lower(s, [A, B, C], simple_mode=True))
testit()

c_x, c_y = C.op.axis
c_xo, c_xi = s[C].split(c_x, factor=factor)
c_yo, c_yi = s[C].split(c_y, factor=factor)
s[C].reorder(c_xo, c_yo, c_xi, c_yi)
print(tvm.lower(s, [A, B, C], simple_mode=True))
testit()

s[C_buf].compute_at(s[C], c_yo)
print(tvm.lower(s, [A, B, C], simple_mode=True))
testit()


def intrin_output_product(n, m):
    a = tvm.placeholder((n,), name='a')
    b = tvm.placeholder((m,), name='b')
    c = tvm.compute((n, m), lambda i, j: a[i] * b[j], name='c')
    Ab = tvm.decl_buffer(a.shape, a.dtype,
                         name="A",
                         offset_factor=1,
                         strides=[1])
    Bb = tvm.decl_buffer(b.shape, b.dtype,
                         name="B",
                         offset_factor=1,
                         strides=[1])
    Cb = tvm.decl_buffer(c.shape, c.dtype,
                         name="C",
                         offset_factor=1,
                         strides=[tvm.var("s1"), 1])

    def intrin_func(ins, outs):

        aa, bb = ins
        cc, = outs

        def _body():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("int32", "outer_product_update",
                                    cc.access_ptr("w"),
                                    aa.access_ptr("r"),
                                    bb.access_ptr("r"),
                                    n, m, cc.strides[0]))
            return ib.get()

        def _reduce_reset():
            ib = tvm.ir_builder.create()
            ib.emit(tvm.call_extern("int32", "outer_product_reset",
                                    cc.access_ptr("w"),
                                    n, m, cc.strides[0]))
            return ib.get()

        def _reduce_update():
            return _body()
        return _body(), _reduce_reset(), _reduce_update()

    with tvm.build_config(offset_factor=1):
        return tvm.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def outer_product_impl():
    cc_code = """
      extern "C" int outer_product_update(float *cc, float *aa, float *bb, int n, int m, int stride) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cc[i * stride + j] += aa[i] * bb[j];
            }
        }
        return 0;
      }
      extern "C" int outer_product_reset(float *cc, int n, int m, int stride) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cc[i * stride + j] = 0;
            }
        }
        return 0;
      }
    """
    from tvm.contrib import util, clang
    temp = util.tempdir()
    return clang.create_llvm(cc_code, output=temp.relpath("temp.ll"))


s[C_buf].tensorize(xi, intrin_output_product(factor, factor))
s[C_buf].pragma(z, "import_llvm", outer_product_impl())
print(tvm.lower(s, [A, B, C], simple_mode=True))

testit()
