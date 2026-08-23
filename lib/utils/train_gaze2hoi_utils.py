import os
import os.path as osp
import shutil
import subprocess
import sys


def setup_pointnet2_sys_path(project_root):
    # Avoid permission issues on ~/.cache/torch_extensions in restricted envs.
    torch_ext_dir = osp.join(project_root, ".torch_extensions")
    os.makedirs(torch_ext_dir, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", torch_ext_dir)
    # Ensure ninja in the current Python env is discoverable for JIT extension builds.
    py_bin_dir = osp.dirname(sys.executable)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if py_bin_dir not in path_entries:
        os.environ["PATH"] = py_bin_dir + os.pathsep + os.environ.get("PATH", "")
    # Make PointNet++ JIT build more tolerant across CUDA toolchain mismatches.
    nvcc_flags = os.environ.get("TORCH_NVCC_FLAGS", "")
    extra_flags = "-allow-unsupported-compiler -DTHRUST_IGNORE_CUB_VERSION_CHECK"
    if extra_flags not in nvcc_flags:
        os.environ["TORCH_NVCC_FLAGS"] = (nvcc_flags + " " + extra_flags).strip()

    pointnet2_root = osp.join(project_root, "Pointnet2_PyTorch")
    if pointnet2_root not in sys.path:
        sys.path.insert(0, pointnet2_root)
    pointnet2_ops_root = osp.join(pointnet2_root, "pointnet2_ops_lib")
    if pointnet2_ops_root not in sys.path:
        sys.path.insert(0, pointnet2_ops_root)


class TeeIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def get_current_git_branch(project_root):
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def save_code_snapshot(project_root, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    exclude_dirs = {
        osp.abspath(dest_dir),
        osp.join(project_root, "outputs"),
    }
    for root, _, files in os.walk(project_root):
        abs_root = osp.abspath(root)
        if any(abs_root.startswith(excl) for excl in exclude_dirs):
            continue
        for fname in files:
            if not (fname.endswith(".py") or fname.endswith(".yaml")):
                continue
            src_path = osp.join(root, fname)
            rel_path = osp.relpath(src_path, project_root)
            dst_path = osp.join(dest_dir, rel_path)
            os.makedirs(osp.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
