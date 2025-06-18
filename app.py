import modal
from src.puzzles import puzzle_11

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install(
        "cupy-cuda12x",
        "numba",
        "git+https://github.com/chalk-diagrams/planar",
        "git+https://github.com/danoneata/chalk@srush-patch-1",
    )
    .add_local_python_source("src.lib", "src.puzzles")
    .add_local_file("robot.png", remote_path="/root/robot.png")
)
app = modal.App("gpu-puzzles", image=image)


@app.function(gpu="T4", max_containers=1)
def modal_wrapper():
    puzzle_11()


@app.local_entrypoint()
def main():
    print("hello world!")
    print(modal_wrapper.remote())
