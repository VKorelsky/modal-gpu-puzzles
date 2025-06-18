import numba
import cupy as np
import warnings
from src.lib import CudaProblem, Coord

warnings.filterwarnings(action="ignore", category=numba.NumbaPerformanceWarning, module="numba")


def puzzle_1():
    def map_spec(a):
        return a + 10

    def map_test(cuda):
        def call(out, a) -> None:
            local_i = cuda.threadIdx.x
            # FILL ME IN (roughly 1 lines)
            out[local_i] = a[local_i] + 10

        return call

    SIZE = 4
    out = np.zeros((SIZE,))
    a = np.arange(SIZE)
    problem = CudaProblem("Map", map_test, [a], out, threadsperblock=Coord(SIZE, 1), spec=map_spec)
    problem.show()
    problem.check()


def puzzle_11():
    def conv_spec(a, b):
        out = np.zeros(*a.shape)
        len = b.shape[0]
        for i in range(a.shape[0]):
            out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
        return out

    MAX_CONV = 4
    TPB = 8
    TPB_MAX_CONV = TPB + MAX_CONV

    def conv_test(cuda):
        def call(out, a, b, a_size, b_size) -> None:
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            local_i = cuda.threadIdx.x

            # FILL ME IN (roughly 17 lines)

        return call

    #### Test 1 ####
    SIZE = 6
    CONV = 3
    out = np.zeros(SIZE)
    a = np.arange(SIZE)
    b = np.arange(CONV)
    problem = CudaProblem(
        "1D Conv (Simple)",
        conv_test,
        [a, b],
        out,
        [SIZE, CONV],
        Coord(1, 1),
        Coord(TPB, 1),
        spec=conv_spec,
    )

    problem.show()
    problem.check()

    #### Test 2 ####
    out = np.zeros(15)
    a = np.arange(15)
    b = np.arange(4)
    problem = CudaProblem(
        "1D Conv (Full)",
        conv_test,
        [a, b],
        out,
        [15, 4],
        Coord(2, 1),
        Coord(TPB, 1),
        spec=conv_spec,
    )
    problem.show()
    problem.check()
