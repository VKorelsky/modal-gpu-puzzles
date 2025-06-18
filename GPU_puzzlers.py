import modal

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "cupy-cuda12x",
        "numba",
        "git+https://github.com/chalk-diagrams/planar",
        "git+https://github.com/danoneata/chalk@srush-patch-1",
    )
    .add_local_file("lib.py", remote_path="/root/lib.py")
    .add_local_file("robot.png", remote_path="/root/robot.png")
)
app = modal.App("gpu-puzzles", image=image)

with image.imports():
    import numba
    import cupy as np
    import warnings
    from lib import CudaProblem, Coord

    warnings.filterwarnings(action="ignore", category=numba.NumbaPerformanceWarning, module="numba")

###########################################################################################################################################################################
# ## Puzzle 11 - 1D Convolution
#
# Implement a kernel that computes a 1D convolution between `a` and `b` and stores it in `out`.
# You need to handle the general case. You only need 2 global reads and 1 global write per thread.


# +
@app.function(gpu="T4", max_containers=1)
def conv_one_puzzle():
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

    # +
    problem.check()
    # -

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


@app.local_entrypoint()
def main():
    print("hello world!")
    print(conv_one_puzzle.remote())


# ###########################################################################################################################################################################

# # ## Puzzle 12 - Prefix Sum
# #
# # Implement a kernel that computes a sum over `a` and stores it in `out`.
# # If the size of `a` is greater than the block size, only store the sum of
# # each block.

# # We will do this using the [parallel prefix sum](https://en.wikipedia.org/wiki/Prefix_sum) algorithm in shared memory.
# # That is, each step of the algorithm should sum together half the remaining numbers.
# # Follow this diagram:

# # ![](https://user-images.githubusercontent.com/35882/178757889-1c269623-93af-4a2e-a7e9-22cd55a42e38.png)

# # +
# TPB = 8


# def sum_spec(a):
#     out = np.zeros((a.shape[0] + TPB - 1) // TPB)
#     for j, i in enumerate(range(0, a.shape[-1], TPB)):
#         out[j] = a[i : i + TPB].sum()
#     return out


# def sum_test(cuda):
#     def call(out, a, size: int) -> None:
#         cache = cuda.shared.array(TPB, numba.float32)
#         i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#         local_i = cuda.threadIdx.x
#         # FILL ME IN (roughly 12 lines)

#     return call


# # Test 1

# SIZE = 8
# out = np.zeros(1)
# inp = np.arange(SIZE)
# problem = CudaProblem(
#     "Sum (Simple)",
#     sum_test,
#     [inp],
#     out,
#     [SIZE],
#     Coord(1, 1),
#     Coord(TPB, 1),
#     spec=sum_spec,
# )
# problem.show()

# # +
# problem.check()
# # -

# # Test 2

# # +
# SIZE = 15
# out = np.zeros(2)
# inp = np.arange(SIZE)
# problem = CudaProblem(
#     "Sum (Full)",
#     sum_test,
#     [inp],
#     out,
#     [SIZE],
#     Coord(2, 1),
#     Coord(TPB, 1),
#     spec=sum_spec,
# )
# problem.show()

# # +
# problem.check()
# # -

# # ## Puzzle 13 - Axis Sum
# #
# # Implement a kernel that computes a sum over each column of `a` and stores it in `out`.

# # +
# TPB = 8


# def sum_spec(a):
#     out = np.zeros((a.shape[0], (a.shape[1] + TPB - 1) // TPB))
#     for j, i in enumerate(range(0, a.shape[-1], TPB)):
#         out[..., j] = a[..., i : i + TPB].sum(-1)
#     return out


# def axis_sum_test(cuda):
#     def call(out, a, size: int) -> None:
#         cache = cuda.shared.array(TPB, numba.float32)
#         i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#         local_i = cuda.threadIdx.x
#         batch = cuda.blockIdx.y
#         # FILL ME IN (roughly 12 lines)

#     return call


# BATCH = 4
# SIZE = 6
# out = np.zeros((BATCH, 1))
# inp = np.arange(BATCH * SIZE).reshape((BATCH, SIZE))
# problem = CudaProblem(
#     "Axis Sum",
#     axis_sum_test,
#     [inp],
#     out,
#     [SIZE],
#     Coord(1, BATCH),
#     Coord(TPB, 1),
#     spec=sum_spec,
# )
# problem.show()

# # +
# problem.check()
# # -

# # ## Puzzle 14 - Matrix Multiply!
# #
# # Implement a kernel that multiplies square matrices `a` and `b` and
# # stores the result in `out`.
# #
# # *Tip: The most efficient algorithm here will copy a block into
# #  shared memory before computing each of the individual row-column
# #  dot products. This is easy to do if the matrix fits in shared
# #  memory.  Do that case first. Then update your code to compute
# #  a partial dot-product and iteratively move the part you
# #  copied into shared memory.* You should be able to do the hard case
# #  in 6 global reads.


# # +
# def matmul_spec(a, b):
#     return a @ b


# TPB = 3


# def mm_oneblock_test(cuda):
#     def call(out, a, b, size: int) -> None:
#         a_shared = cuda.shared.array((TPB, TPB), numba.float32)
#         b_shared = cuda.shared.array((TPB, TPB), numba.float32)

#         i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
#         j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
#         local_i = cuda.threadIdx.x
#         local_j = cuda.threadIdx.y
#         # FILL ME IN (roughly 14 lines)

#     return call


# # Test 1

# SIZE = 2
# out = np.zeros((SIZE, SIZE))
# inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
# inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

# problem = CudaProblem(
#     "Matmul (Simple)",
#     mm_oneblock_test,
#     [inp1, inp2],
#     out,
#     [SIZE],
#     Coord(1, 1),
#     Coord(TPB, TPB),
#     spec=matmul_spec,
# )
# problem.show(sparse=True)

# # +
# problem.check()
# # -

# # Test 2

# # +
# SIZE = 8
# out = np.zeros((SIZE, SIZE))
# inp1 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE))
# inp2 = np.arange(SIZE * SIZE).reshape((SIZE, SIZE)).T

# problem = CudaProblem(
#     "Matmul (Full)",
#     mm_oneblock_test,
#     [inp1, inp2],
#     out,
#     [SIZE],
#     Coord(3, 3),
#     Coord(TPB, TPB),
#     spec=matmul_spec,
# )
# problem.show(sparse=True)
# # -

# # +
# problem.check()
# # -
