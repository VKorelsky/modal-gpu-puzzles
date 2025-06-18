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


def puzzle_2():
    def zip_spec(a, b):
        return a + b

    def zip_test(cuda):
        def call(out, a, b) -> None:
            local_i = cuda.threadIdx.x
            # FILL ME IN (roughly 1 lines)
            out[local_i] = a[local_i] + b[local_i]

        return call

    SIZE = 4
    out = np.zeros((SIZE,))
    a = np.arange(SIZE)
    b = np.arange(SIZE)
    problem = CudaProblem("Zip", zip_test, [a, b], out, threadsperblock=Coord(SIZE, 1), spec=zip_spec)
    problem.show()
    problem.check()


def puzzle_3():
    def map_spec(a):
        return a + 10

    def map_guard_test(cuda):
        def call(out, a, size) -> None:
            local_i = cuda.threadIdx.x
            # FILL ME IN (roughly 2 lines)

            if local_i >= size:
                return

            out[local_i] = a[local_i] + 10

        return call

    SIZE = 4
    out = np.zeros((SIZE,))
    a = np.arange(SIZE)
    problem = CudaProblem(
        "Guard",
        map_guard_test,
        [a],
        out,
        (SIZE,),
        threadsperblock=Coord(8, 1),
        spec=map_spec,
    )
    problem.show()
    problem.check()


def puzzle_4():
    def map_spec(a):
        return a + 10

    def map_2D_test(cuda):
        def call(out, a, size) -> None:
            local_i = cuda.threadIdx.x
            local_j = cuda.threadIdx.y
            # FILL ME IN (roughly 2 lines)
            if local_j >= size or local_i >= size:
                return

            out[local_i, local_j] = a[local_i, local_j] + 10

        return call

    SIZE = 2
    out = np.zeros((SIZE, SIZE))
    a = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
    problem = CudaProblem("Map 2D", map_2D_test, [a], out, (SIZE,), threadsperblock=Coord(3, 3), spec=map_spec)
    problem.show()
    problem.check()


def puzzle_5():
    def zip_spec(a, b):
        return a + b

    def broadcast_test(cuda):
        def call(out, a, b, size) -> None:
            local_i = cuda.threadIdx.x
            local_j = cuda.threadIdx.y
            # FILL ME IN (roughly 2 lines)

            if local_i >= size or local_j >= size:
                return

            out[local_i, local_j] = a[local_i, 0] + b[0, local_j]

        return call

    SIZE = 2
    out = np.zeros((SIZE, SIZE))
    a = np.arange(SIZE).reshape(SIZE, 1)
    b = np.arange(SIZE).reshape(1, SIZE)
    problem = CudaProblem(
        "Broadcast",
        broadcast_test,
        [a, b],
        out,
        (SIZE,),
        threadsperblock=Coord(3, 3),
        spec=zip_spec,
    )
    problem.show()
    problem.check()


def puzzle_6():
    def map_spec(a):
        return a + 10

    def map_block_test(cuda):
        def call(out, a, size) -> None:
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            # FILL ME IN (roughly 2 lines)

        return call

    SIZE = 9
    out = np.zeros((SIZE,))
    a = np.arange(SIZE)
    problem = CudaProblem(
        "Blocks",
        map_block_test,
        [a],
        out,
        (SIZE,),
        threadsperblock=Coord(4, 1),
        blockspergrid=Coord(3, 1),
        spec=map_spec,
    )
    problem.show()
    problem.check()


def puzzle_7():
    def map_spec(a):
        return a + 10

    def map_block2D_test(cuda):
        def call(out, a, size) -> None:
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            # FILL ME IN (roughly 4 lines)

        return call

    SIZE = 5
    out = np.zeros((SIZE, SIZE))
    a = np.ones((SIZE, SIZE))

    problem = CudaProblem(
        "Blocks 2D",
        map_block2D_test,
        [a],
        out,
        (SIZE,),
        threadsperblock=Coord(3, 3),
        blockspergrid=Coord(2, 2),
        spec=map_spec,
    )
    problem.show()
    problem.check()


def puzzle_8():
    def map_spec(a):
        return a + 10

    TPB = 4

    def shared_test(cuda):
        def call(out, a, size) -> None:
            shared = cuda.shared.array(TPB, numba.float32)
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            local_i = cuda.threadIdx.x

            if i < size:
                shared[local_i] = a[i]
                cuda.syncthreads()

            # FILL ME IN (roughly 2 lines)

        return call

    SIZE = 8
    out = np.zeros(SIZE)
    a = np.ones(SIZE)
    problem = CudaProblem(
        "Shared",
        shared_test,
        [a],
        out,
        (SIZE,),
        threadsperblock=Coord(TPB, 1),
        blockspergrid=Coord(2, 1),
        spec=map_spec,
    )
    problem.show()
    problem.check()


def puzzle_9():
    def pool_spec(a):
        out = np.zeros(*a.shape)
        for i in range(a.shape[0]):
            out[i] = a[max(i - 2, 0) : i + 1].sum()
        return out

    TPB = 8

    def pool_test(cuda):
        def call(out, a, size) -> None:
            shared = cuda.shared.array(TPB, numba.float32)
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            local_i = cuda.threadIdx.x
            # FILL ME IN (roughly 8 lines)

        return call

    SIZE = 8
    out = np.zeros(SIZE)
    a = np.arange(SIZE)
    problem = CudaProblem(
        "Pooling",
        pool_test,
        [a],
        out,
        (SIZE,),
        threadsperblock=Coord(TPB, 1),
        blockspergrid=Coord(1, 1),
        spec=pool_spec,
    )
    problem.show()
    problem.check()


def puzzle_10():
    def dot_spec(a, b):
        return a @ b

    TPB = 8

    def dot_test(cuda):
        def call(out, a, b, size) -> None:
            shared = cuda.shared.array(TPB, numba.float32)

            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            local_i = cuda.threadIdx.x
            # FILL ME IN (roughly 9 lines)

        return call

    SIZE = 8
    out = np.zeros(1)
    a = np.arange(SIZE)
    b = np.arange(SIZE)
    problem = CudaProblem(
        "Dot",
        dot_test,
        [a, b],
        out,
        (SIZE,),
        threadsperblock=Coord(SIZE, 1),
        blockspergrid=Coord(1, 1),
        spec=dot_spec,
    )
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
        (SIZE, CONV),
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
        (15, 4),
        Coord(2, 1),
        Coord(TPB, 1),
        spec=conv_spec,
    )
    problem.show()
    problem.check()


def puzzle_12():
    TPB = 8

    def sum_spec(a):
        out = np.zeros((a.shape[0] + TPB - 1) // TPB)
        for j, i in enumerate(range(0, a.shape[-1], TPB)):
            out[j] = a[i : i + TPB].sum()
        return out

    def sum_test(cuda):
        def call(out, a, size: int) -> None:
            cache = cuda.shared.array(TPB, numba.float32)
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            local_i = cuda.threadIdx.x
            # FILL ME IN (roughly 12 lines)

        return call

    #### Test 1 ####
    SIZE = 8
    out = np.zeros(1)
    inp = np.arange(SIZE)
    problem = CudaProblem(
        "Sum (Simple)",
        sum_test,
        [inp],
        out,
        (SIZE,),
        Coord(1, 1),
        Coord(TPB, 1),
        spec=sum_spec,
    )
    problem.show()
    problem.check()

    #### Test 2 ####
    SIZE = 15
    out = np.zeros(2)
    inp = np.arange(SIZE)
    problem = CudaProblem(
        "Sum (Full)",
        sum_test,
        [inp],
        out,
        (SIZE,),
        Coord(2, 1),
        Coord(TPB, 1),
        spec=sum_spec,
    )
    problem.show()
    problem.check()


def puzzle_13():
    TPB = 8

    def sum_spec(a):
        out = np.zeros((a.shape[0], (a.shape[1] + TPB - 1) // TPB))
        for j, i in enumerate(range(0, a.shape[-1], TPB)):
            out[..., j] = a[..., i : i + TPB].sum(-1)
        return out

    def axis_sum_test(cuda):
        def call(out, a, size: int) -> None:
            cache = cuda.shared.array(TPB, numba.float32)
            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            local_i = cuda.threadIdx.x
            batch = cuda.blockIdx.y
            # FILL ME IN (roughly 12 lines)

        return call

    BATCH = 4
    SIZE = 6
    out = np.zeros((BATCH, 1))
    inp = np.arange(BATCH * SIZE).reshape((BATCH, SIZE))
    problem = CudaProblem(
        "Axis Sum",
        axis_sum_test,
        [inp],
        out,
        (SIZE,),
        Coord(1, BATCH),
        Coord(TPB, 1),
        spec=sum_spec,
    )
    problem.show()
    problem.check()


def puzzle_14():
    def matmul_spec(a, b):
        return a @ b

    TPB = 3

    def mm_oneblock_test(cuda):
        def call(out, a, b, size: int) -> None:
            a_shared = cuda.shared.array((TPB, TPB), numba.float32)
            b_shared = cuda.shared.array((TPB, TPB), numba.float32)

            i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
            j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
            local_i = cuda.threadIdx.x
            local_j = cuda.threadIdx.y
            # FILL ME IN (roughly 14 lines)

        return call

    #### Test 1 ####
    SIZE = 2
    out = np.zeros((SIZE, SIZE))
    inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
    inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

    problem = CudaProblem(
        "Matmul (Simple)",
        mm_oneblock_test,
        [inp1, inp2],
        out,
        (SIZE,),
        Coord(1, 1),
        Coord(TPB, TPB),
        spec=matmul_spec,
    )
    problem.show(sparse=True)
    problem.check()

    #### Test 2 ####
    SIZE = 8
    out = np.zeros((SIZE, SIZE))
    inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
    inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

    problem = CudaProblem(
        "Matmul (Full)",
        mm_oneblock_test,
        [inp1, inp2],
        out,
        (SIZE,),
        Coord(3, 3),
        Coord(TPB, TPB),
        spec=matmul_spec,
    )
    problem.show(sparse=True)
    problem.check()


def run_puzzle(i: int):
    match i:
        case 1:
            puzzle_1()
        case 2:
            puzzle_2()
        case 3:
            puzzle_3()
        case 4:
            puzzle_4()
        case 5:
            puzzle_5()
        case 6:
            puzzle_6()
        case 7:
            puzzle_7()
        case 8:
            puzzle_8()
        case 9:
            puzzle_9()
        case 10:
            puzzle_10()
        case 11:
            puzzle_11()
        case 12:
            puzzle_12()
        case 13:
            puzzle_13()
        case 14:
            puzzle_14()
        case _:
            raise ValueError(f"Invalid puzzle number: {i}")
