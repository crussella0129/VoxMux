"""install.py: install everything VoxMux needs, then check that it worked."""

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
    """Run one pip install per library, stopping at the first failure."""
    for step in STEPS:
        subprocess.run(PIP + step, check=True)


def copy_cuda_files():
    """Copy the three NVIDIA files llama.cpp needs from PyTorch."""
    site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
    source = site_packages / "torch" / "lib"
    target = site_packages / "llama_cpp" / "lib"
    for name in CUDA_FILES:
        shutil.copy(source / name, target)


def check():
    """Import everything in a fresh Python and print what it finds."""
    subprocess.run([sys.executable, "-c", CHECK], check=True)


def main():
    """Install, copy, then check."""
    install_libraries()
    copy_cuda_files()
    check()


if __name__ == "__main__":
    main()
