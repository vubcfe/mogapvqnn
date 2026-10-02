"""Datasets, binary task construction and amplitude encoding.

* MNIST / Fashion-MNIST are downloaded once as IDX files (no torch needed).
* EuroSAT (RGB, 64x64) is read from an ``ImageFolder``-style directory
  (``<root>/<ClassName>/*.jpg``); it is downloaded from Zenodo if missing.

A binary task draws class-balanced train/test samples with the run seed,
downsamples every image to sqrt(2^n) x sqrt(2^n) pixels, flattens it and
l2-normalises it into an n-qubit amplitude-encoded state (paper Eq. 3).
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tempfile
import struct
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

IDX_MIRRORS = {
    "mnist": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
    ],
    "fashion_mnist": [
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    ],
}
IDX_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
EUROSAT_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip"
FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
# Fixed seed for the EuroSAT train/test pool split (EuroSAT has no official
# split); independent of the run seed so the test pool never changes.
EUROSAT_POOL_SEED = 20240601


def default_data_dir() -> Path:
    return Path(os.environ.get("MOGAPVQNN_DATA", Path(__file__).resolve().parents[2] / "data"))


def _download(url: str, dest: Path) -> None:
    """Download to a unique temporary file, then atomically move into place,
    so concurrent processes can never observe or produce a partial file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part", dir=dest.parent)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mogapvqnn"})
        with urllib.request.urlopen(req, timeout=120) as r, os.fdopen(fd, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, dest)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    zero, dtype, ndim = struct.unpack(">HBB", data[:4])
    if zero != 0 or dtype != 0x08:
        raise ValueError(f"{path} is not an unsigned-byte IDX file")
    shape = struct.unpack(">" + "I" * ndim, data[4 : 4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_idx_dataset(name: str, root: Path | None = None):
    """Return (X_train, y_train, X_test, y_test) as uint8 arrays."""
    if name not in IDX_MIRRORS:
        raise ValueError(f"unknown IDX dataset {name!r}; expected one of {sorted(IDX_MIRRORS)}")
    root = Path(root or default_data_dir()) / "raw" / name
    arrays = {}
    for key, fname in IDX_FILES.items():
        path = root / fname
        if path.exists():
            try:
                arrays[key] = _read_idx(path)
                continue
            except Exception:            # corrupt / HTML error page: fetch again
                path.unlink(missing_ok=True)
        errors = []
        for base in IDX_MIRRORS[name]:
            try:
                _download(base + fname, path)
                arrays[key] = _read_idx(path)
                break
            except Exception as e:      # try the next mirror
                path.unlink(missing_ok=True)
                errors.append(f"{base}: {e}")
        else:
            raise RuntimeError(f"could not download {fname}:\n" + "\n".join(errors))
    return arrays["train_images"], arrays["train_labels"], arrays["test_images"], arrays["test_labels"]


def _eurosat_root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("EUROSAT_ROOT")
    if env:
        return Path(env)
    return default_data_dir() / "raw" / "EuroSAT_RGB"


def _find_class_root(base: Path, classes: list[str]) -> Path | None:
    """Directory under ``base`` (searched two levels deep) containing all class folders."""
    for cand in [base, *sorted(p for p in base.glob("*") if p.is_dir()),
                 *sorted(p for p in base.glob("*/*") if p.is_dir())]:
        if all((cand / c).is_dir() for c in classes):
            return cand
    return None


def load_eurosat(classes: list[str], root: Path | None = None, cache_dir: Path | None = None,
                 data_dir: Path | None = None):
    """Return (images uint8 (N,64,64,3), labels int in {0,1})."""
    data_dir = Path(data_dir or default_data_dir())
    root = _eurosat_root(root)
    cache_dir = Path(cache_dir or data_dir / "cache")
    tag = hashlib.blake2b(str(root.resolve()).encode(), digest_size=4).hexdigest()
    cache = cache_dir / f"eurosat_{'_'.join(classes)}_{tag}.npz"
    if cache.exists():
        try:
            with np.load(cache) as z:
                return z["X"], z["y"]
        except Exception:                # truncated cache: rebuild
            cache.unlink(missing_ok=True)
    if not all((root / c).is_dir() for c in classes):
        zip_path = data_dir / "raw" / "EuroSAT_RGB.zip"
        if not zip_path.exists():
            _download(EUROSAT_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(zip_path.parent)
        found = _find_class_root(zip_path.parent, classes)
        if found is None:
            raise FileNotFoundError(f"classes {classes} not found after extracting {zip_path}")
        root = found
    X, y = [], []
    for label, cname in enumerate(classes):
        files = sorted((root / cname).glob("*.jpg")) + sorted((root / cname).glob("*.png"))
        if not files:
            raise FileNotFoundError(f"no images for class {cname} under {root}")
        for f in files:
            with Image.open(f) as im:
                X.append(np.asarray(im.convert("RGB")))
            y.append(label)
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=cache.name + ".", suffix=".part", dir=cache_dir)
    with os.fdopen(fd, "wb") as f:      # atomic write: safe under parallel runs
        np.savez_compressed(f, X=X, y=y)
    os.replace(tmp, cache)
    return X, y


# --------------------------------------------------------------------------- #
# Preprocessing and encoding
# --------------------------------------------------------------------------- #

def image_side(n_qubits: int) -> int:
    if n_qubits % 2:
        raise ValueError("amplitude encoding of square images needs an even qubit count")
    return 2 ** (n_qubits // 2)


def downsample(images: np.ndarray, side: int, method: str = "nearest") -> np.ndarray:
    """uint8 images (N,H,W) or (N,H,W,3) -> float (N, side*side) in [0, 1].
    RGB is converted to grayscale with the ITU-R 601 luma transform."""
    resample = {"nearest": Image.NEAREST, "box": Image.BOX, "bilinear": Image.BILINEAR}[method]
    out = np.empty((len(images), side * side))
    for i, img in enumerate(images):
        im = Image.fromarray(img)
        if im.mode != "L":
            im = im.convert("L")
        out[i] = np.asarray(im.resize((side, side), resample), dtype=float).ravel() / 255.0
    return out


def amplitude_encode(x: np.ndarray) -> np.ndarray:
    """l2-normalise rows; an all-zero image maps to the uniform superposition."""
    x = np.asarray(x, dtype=float)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    zero = norms[:, 0] == 0
    x = np.where(zero[:, None], 1.0, x)
    norms = np.where(zero[:, None], np.sqrt(x.shape[1]), norms)
    return x / norms


@dataclass
class BinaryTask:
    name: str
    n_qubits: int
    X_fit: np.ndarray
    y_fit: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict = field(default_factory=dict)

    def split(self, name: str):
        return getattr(self, f"X_{name}"), getattr(self, f"y_{name}")


def _balanced_sample(y: np.ndarray, n_total: int, rng: np.random.Generator) -> np.ndarray:
    per = [n_total // 2, n_total - n_total // 2]
    idx = []
    for label, k in zip((0, 1), per):
        pool = np.flatnonzero(y == label)
        if len(pool) < k:
            raise ValueError(f"class {label}: requested {k} samples, only {len(pool)} available")
        idx.append(rng.choice(pool, size=k, replace=False))
    idx = np.concatenate(idx)
    rng.shuffle(idx)
    return idx


def _stratified_holdout(y: np.ndarray, frac: float, rng: np.random.Generator):
    hold = []
    for label in (0, 1):
        pool = np.flatnonzero(y == label)
        if len(pool) < 2:
            raise ValueError(f"class {label} needs at least 2 training samples for a fit/val split")
        k = int(np.floor(frac * len(pool) + 0.5))         # round half up
        k = min(max(k, 1), len(pool) - 1)                  # both parts non-empty
        hold.append(rng.choice(pool, size=k, replace=False))
    hold = np.sort(np.concatenate(hold))
    keep = np.setdiff1d(np.arange(len(y)), hold)
    return rng.permutation(keep), rng.permutation(hold)


def make_task(
    dataset: str,
    classes: list,
    n_qubits: int,
    n_train: int,
    n_test: int,
    seed: int,
    val_fraction: float = 0.2,
    resize: str = "nearest",
    data_dir: Path | None = None,
    eurosat_root: Path | None = None,
) -> BinaryTask:
    """Build the amplitude-encoded binary task for one run seed.

    The training sample is split stratified into a fit part (classifier
    training) and a validation part (GA fitness, ensemble selection);
    the test sample is only used for final reporting.
    """
    # independent streams: the test sample does not depend on n_train and the
    # fit/val split does not depend on n_test
    rng_train = np.random.default_rng([seed, 7, 1])
    rng_test = np.random.default_rng([seed, 7, 2])
    rng_split = np.random.default_rng([seed, 7, 3])
    if dataset in ("mnist", "fashion_mnist"):
        Xtr, ytr, Xte, yte = load_idx_dataset(dataset, data_dir)
        a, b = (int(c) for c in classes)
        mtr, mte = np.isin(ytr, [a, b]), np.isin(yte, [a, b])
        Xtr, ytr = Xtr[mtr], (ytr[mtr] == b).astype(np.int64)
        Xte, yte = Xte[mte], (yte[mte] == b).astype(np.int64)
        class_names = [FASHION_LABELS[a], FASHION_LABELS[b]] if dataset == "fashion_mnist" else [str(a), str(b)]
    elif dataset == "eurosat":
        X, y = load_eurosat(list(classes), eurosat_root, data_dir=data_dir)
        perm = np.random.default_rng(EUROSAT_POOL_SEED).permutation(len(y))
        cut = int(0.8 * len(y))
        Xtr, ytr, Xte, yte = X[perm[:cut]], y[perm[:cut]], X[perm[cut:]], y[perm[cut:]]
        class_names = list(classes)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    itr = _balanced_sample(ytr, n_train, rng_train)
    ite = _balanced_sample(yte, n_test, rng_test)
    side = image_side(n_qubits)
    S_tr = amplitude_encode(downsample(Xtr[itr], side, resize))
    S_te = amplitude_encode(downsample(Xte[ite], side, resize))
    y_tr, y_te = ytr[itr], yte[ite]
    fit_idx, val_idx = _stratified_holdout(y_tr, val_fraction, rng_split)
    return BinaryTask(
        name=dataset,
        n_qubits=n_qubits,
        X_fit=S_tr[fit_idx], y_fit=y_tr[fit_idx],
        X_val=S_tr[val_idx], y_val=y_tr[val_idx],
        X_test=S_te, y_test=y_te,
        meta={
            "classes": [str(c) for c in classes],
            "class_names": class_names,
            "image_side": side,
            "resize": resize,
            "n_fit": len(fit_idx), "n_val": len(val_idx), "n_test": len(ite),
        },
    )


def prepare_datasets(configurations: list[dict], data_dir: Path | None = None) -> None:
    """Download / cache every dataset a suite needs, once, before parallel runs start."""
    seen = set()
    for c in configurations:
        key = (c["dataset"], tuple(map(str, c.get("classes", []))))
        if key in seen:
            continue
        seen.add(key)
        if c["dataset"] in IDX_MIRRORS:
            load_idx_dataset(c["dataset"], data_dir)
        elif c["dataset"] == "eurosat":
            load_eurosat(list(c["classes"]), data_dir=data_dir)
        else:
            raise ValueError(f"unknown dataset {c['dataset']!r}")


def synthetic_task(n_qubits: int, n_train: int = 120, n_test: int = 60, seed: int = 0) -> BinaryTask:
    """Small linearly-structured task for tests and smoke runs."""
    rng = np.random.default_rng(seed)
    D = 2**n_qubits
    centers = rng.random((2, D))

    def draw(k):
        y = np.arange(k) % 2
        x = np.abs(centers[y] + 0.35 * rng.normal(size=(k, D)))
        return amplitude_encode(x), y

    S_tr, y_tr = draw(n_train)
    S_te, y_te = draw(n_test)
    fit_idx, val_idx = _stratified_holdout(y_tr, 0.2, rng)
    return BinaryTask(
        "synthetic", n_qubits,
        S_tr[fit_idx], y_tr[fit_idx], S_tr[val_idx], y_tr[val_idx], S_te, y_te,
        meta={"n_fit": len(fit_idx), "n_val": len(val_idx), "n_test": n_test},
    )
