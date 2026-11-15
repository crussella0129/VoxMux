"""Install VoxMux's libraries into the Python running this script, then check them.

Windows and an NVIDIA graphics card only. Safe to run more than once.
"""

import pathlib
import shutil
import subprocess
import sys
import sysconfig

PIP = [sys.executable, "-m", "pip", "install"]
STEPS = [
    ["torch", "--index-url", "https://download.pytorch.org/whl/cu124"],
    ["transformers>=5"],
    [
        "--only-binary=llama-cpp-python",
        # 0.3.37 crashed with an illegal-instruction error on the test machine.
        "llama-cpp-python==0.3.36",
        "--extra-index-url",
        "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    ],
    ["tzdata"],
]
CUDA_FILES = ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"]
CHECK = (
    "import tkinter, llama_cpp, torch, transformers; "
    "print(llama_cpp.__version__, torch.__version__, transformers.__version__, "
    "'cuda:', torch.cuda.is_available(), "
    "'llama gpu:', llama_cpp.llama_supports_gpu_offload())"
)


def install_libraries():
    for step in STEPS:
        subprocess.run(PIP + step, check=True)


def copy_cuda_files():
    # llama.cpp's CUDA build needs these runtime files but does not ship them.
    site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
    source = site_packages / "torch" / "lib"
    target = site_packages / "llama_cpp" / "lib"
    for name in CUDA_FILES:
        shutil.copy(source / name, target)


def check():
    subprocess.run([sys.executable, "-c", CHECK], check=True)


def main():
    install_libraries()
    copy_cuda_files()
    check()


if __name__ == "__main__":
    main()
