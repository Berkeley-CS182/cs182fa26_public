"""Portable setup and submission helpers for CS 182/282A Fall 2026 HW02."""
from pathlib import Path
import ast
import hashlib
import json
import re
import tarfile
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
# Both sources must provide the byte-identical original Python archive.
CIFAR_DOWNLOAD_URLS = (
    'https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz',
    CIFAR_URL,
)
CIFAR_MD5 = 'c58f30108f718f92721af3b95e74349a'


def _md5(path):
    digest = hashlib.md5()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


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
    if archive.is_file() and _md5(archive) == CIFAR_MD5:
        print('Using the already downloaded, verified CIFAR-10 archive.')
    else:
        partial = archive.with_suffix(archive.suffix + '.part')
        failures = []
        for url in CIFAR_DOWNLOAD_URLS:
            print(f'Downloading CIFAR-10 from {url}\n'
                  '(about 170 MB; 60-second connection/read timeout)...', flush=True)
            request = urllib.request.Request(url, headers={'User-Agent': 'CS182-HW02-Fall2026'})
            try:
                with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as output:
                    received = 0
                    next_progress = 10 * 1024 * 1024
                    for chunk in iter(lambda: response.read(1024 * 1024), b''):
                        output.write(chunk)
                        received += len(chunk)
                        if received >= next_progress:
                            print(f'Downloaded {received // (1024 * 1024)} MiB...', flush=True)
                            next_progress += 10 * 1024 * 1024
                if _md5(partial) != CIFAR_MD5:
                    raise ValueError('CIFAR-10 archive checksum mismatch.')
                partial.replace(archive)
                break
            except (OSError, urllib.error.URLError, ValueError) as error:
                failures.append(f'{url}: {error}')
                print(f'Download failed: {error}', flush=True)
        else:
            raise RuntimeError(
                'Both CIFAR-10 download sources failed or stalled. Rerun this cell, or download '
                f'{CIFAR_URL} in a browser, save it as {archive}, and rerun. '
                'A manually supplied archive is verified before extraction.\n' + '\n'.join(failures)
            )
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


def _validate_saved_notebook(path):
    """Catch the unchanged on-disk starter without executing or grading code."""
    try:
        notebook = json.loads(path.read_text(encoding='utf-8'))
        cells = notebook['cells']
        if not isinstance(cells, list):
            raise ValueError('Notebook cells must be a list.')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f'Cannot read the saved notebook {path}. '
                         'Download your completed notebook as .ipynb, replace this file, and rerun packaging.') from error

    variables = ('lr', 'num_epochs', 'batch_size', 'lr_decay')

    def bind(target, value, state):
        if isinstance(target, ast.Name) and target.id in state:
            state[target.id] = isinstance(value, ast.Constant) and value.value is None
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = value.elts if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts) else [None] * len(target.elts)
            for item, item_value in zip(target.elts, values):
                bind(item, item_value, state)

    def assigned_none(statements, initial):
        state = initial.copy()
        for statement in statements:
            if isinstance(statement, ast.Assign):
                for target in statement.targets:
                    bind(target, statement.value, state)
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                bind(statement.target, statement.value, state)
            elif isinstance(statement, ast.AugAssign):
                bind(statement.target, None, state)
            elif isinstance(statement, ast.If):
                left = assigned_none(statement.body, state)
                right = assigned_none(statement.orelse, state)
                state = {name: left[name] and right[name] for name in variables}
            elif isinstance(statement, (ast.For, ast.While, ast.Try, ast.With)):
                # Complex control flow is not graded here. Do not reject a
                # potentially valid reassignment merely because it is conditional.
                for node in ast.walk(statement):
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in state:
                        state[node.id] = False
        return state

    # Follow assignments across cells so moving/splitting the tuning section
    # does not force a particular notebook layout. Unknown values are allowed.
    state = dict.fromkeys(variables, False)
    tuning_names = set()
    possible_writes = set()
    for cell in cells:
        if not isinstance(cell, dict) or cell.get('cell_type') != 'code':
            continue
        source = cell.get('source', '')
        source = source if isinstance(source, str) else ''.join(source)
        tuning_names.update(name for name in ('best_model', 'best_solver') if name in source)
        # Magics may assign notebook globals. Preserve that uncertainty when
        # removing their non-Python syntax, rather than discarding the write.
        lines = []
        for line in source.splitlines():
            if line.lstrip().startswith(('%', '!')):
                possible_writes.update(name for name in variables if re.search(r'\b' + name + r'\b', line))
            else:
                lines.append(line)
        source = '\n'.join(lines)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # IPython supports syntax outside ordinary Python. This narrow
            # stale-starter check must not become a notebook syntax grader.
            state.update((name, False) for name in variables if name in source)
            continue
        # A helper function may write declared globals when called elsewhere.
        # We do not execute calls or infer their ordering across notebook cells.
        possible_writes.update(name for node in ast.walk(tree) if isinstance(node, ast.Global)
                               for name in node.names if name in variables)
        state = assigned_none(tree.body, state)
    unfinished = [name for name in variables if state[name] and name not in possible_writes]
    if tuning_names == {'best_model', 'best_solver'} and unfinished:
        raise ValueError(f'The saved notebook {path} still has None model-tuning placeholders: '
                         + ', '.join(unfinished) + '. Download your completed Colab notebook as .ipynb, '
                         'upload it to replace this exact file, and rerun packaging. '
                         'Locally, save your completed notebook to this path.')


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
    _validate_saved_notebook(files[0])
    print('Packaging saved notebook:', files[0])
    output = ROOT / 'cs182hw2_fa26_submission.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for path in files:
            handle.write(path, path.relative_to(ROOT))
    return output


if __name__ == '__main__':
    download_cifar10()
