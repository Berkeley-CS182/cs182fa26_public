"""Portable setup and submission helpers for CS 182/282A Fall 2026 HW02."""
from pathlib import Path
import hashlib
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_MD5 = 'c58f30108f718f92721af3b95e74349a'


def download_cifar10():
    """Download the original CIFAR-10 Python archive once and verify it."""
    datasets = ROOT / 'deeplearning' / 'datasets'
    target = datasets / 'cifar-10-batches-py'
    required = [target / f'data_batch_{i}' for i in range(1, 6)]
    required.append(target / 'test_batch')
    if all(path.is_file() for path in required):
        print('CIFAR-10 already present:', target)
        return target
    datasets.mkdir(parents=True, exist_ok=True)
    archive = datasets / 'cifar-10-python.tar.gz'
    print('Downloading CIFAR-10 (about 170 MB)...')
    urllib.request.urlretrieve(CIFAR_URL, archive)
    digest = hashlib.md5()
    with archive.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != CIFAR_MD5:
        raise ValueError('CIFAR-10 archive checksum mismatch. Rerun to download it again.')
    # Only ordinary files/directories within the dataset directory may be extracted.
    with tarfile.open(archive, 'r:gz') as handle:
        for member in handle.getmembers():
            destination = (datasets / member.name).resolve()
            if datasets.resolve() not in destination.parents or not (member.isfile() or member.isdir()):
                raise ValueError('Unexpected path or link in CIFAR-10 archive.')
        handle.extractall(datasets)
    archive.unlink()
    if not all(path.is_file() for path in required):
        raise FileNotFoundError('The downloaded archive did not contain the expected CIFAR-10 batches.')
    return target


def build_submission(include_rmsprop=False):
    """Package only assignment sources and the required experiment artifacts."""
    log_names = [f'optimizer_experiment_{rule}.npz' for rule in ('sgd', 'sgd_momentum', 'adam')]
    if include_rmsprop:
        log_names.append('optimizer_experiment_rmsprop.npz')
    log_names += [f'sgd_momentum_compare_{rule}_{seed}.npz'
                  for rule in ('sgd', 'sgd_momentum') for seed in (100, 200, 300)]
    log_names += [f'initialization_experiment_{name}.npz' for name in ('he', 'random', 'zero')]
    log_names += [f'w_stds_{name}.json' for name in ('he', 'random', 'zero')]
    log_names += ['best_fc_model.npz', 'results.json']
    files = [ROOT / 'hw2_optimizer_init.ipynb', ROOT / 'assignment_utils.py']
    files += sorted((ROOT / 'deeplearning').rglob('*.py'))
    files += [ROOT / 'submission_logs' / name for name in log_names]
    missing = [str(path.relative_to(ROOT)) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError('Complete and save the assignment before packaging. Missing: ' + ', '.join(missing))
    output = ROOT / 'cs182hw2_fa26_submission.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, path.relative_to(ROOT))
    return output


if __name__ == '__main__':
    download_cifar10()
