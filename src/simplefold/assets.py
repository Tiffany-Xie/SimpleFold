"""Official PyTorch weights and the reference CCD, stored once per project."""

import fcntl
import hashlib
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from urllib.request import urlopen

APPLE = 'https://ml-site.cdn-apple.com/models/simplefold/'
META = 'https://dl.fbaipublicfiles.com/fair-esm/'
ESM = 'esm2_t36_3B_UR50D'
CCD_URL = 'https://huggingface.co/boltz-community/boltz-1/resolve/7c1d83b779e4c65ecc37dfdf0c6b2788076f31e1/ccd.pkl'
SCALES = ('100M', '360M', '700M', '1.6B')

# SHA-256 of the released bytes. Sizes allow fast offline reuse on compute nodes.
RELEASES = {
    'simplefold_100M.ckpt': (386772550, '4cd0b8a0b317a6ab8634444fffd78ce84cfd49c20fe927b83c76c36fda5f54bd'),
    'simplefold_360M.ckpt': (1454881694, '517338ec36b10ecc774f36b592ffe0fee6a24fa5c7d2fcfa3e3009282d48a49b'),
    'simplefold_700M.ckpt': (2764152946, '77a770e2dd1695397ab5943d5b31c99b9039b8a35ed86db12f2b2db1be9db7f8'),
    'simplefold_1.6B.ckpt': (6354525226, 'aaac2d73dcc59c61153c58a1d56e74a8ada9d6057d67000f7836f3c87325312b'),
    'simplefold_1.1B.ckpt': (4469540038, ''),
    'simplefold_3B.ckpt': (11460066894, ''),
    ESM + '.pt': (5678116398, '7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500'),
    ESM + '-contact-regression.pt': (6759, '4da500eab246481dc9c8c95bc7b1d02f2803d761c380b0e95186d4a07d0fc84e'),
    'plddt.ckpt': (462812900, ''),
    'ccd.pkl': (345859128, '2d3b2f03a3c5665944adba51e33263511e51b21c9cd05d902f9c4b7c1e58d2f4'),
}


def url_for(name):
    if name == 'plddt.ckpt':
        return APPLE + 'plddt_module_1.6B.ckpt'
    if name == 'ccd.pkl':
        return CCD_URL
    if name.startswith('esm2_'):
        folder = 'regression' if name.endswith('-regression.pt') else 'models'
        return f'{META}{folder}/{name}'
    return APPLE + name


@contextmanager
def cache_lock(path):
    locks = path.parent / '.locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / path.name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def resource(root, name, *, verify=False):
    size, digest = RELEASES[name]
    path = Path(root) / name
    def ready():
        if not path.is_file() or path.stat().st_size != size:
            return False
        if verify and digest:
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(4 << 20), b''):
                    h.update(chunk)
            if h.hexdigest() != digest:
                raise ValueError(f'Checksum mismatch for {path}')
        return True

    if ready():
        return path
    with cache_lock(path):
        if ready():
            return path
        part = path.with_suffix(path.suffix + '.part')
        h = hashlib.sha256()
        try:
            with urlopen(url_for(name), timeout=120) as src, part.open('wb') as out:
                for chunk in iter(lambda: src.read(4 << 20), b''):
                    out.write(chunk)
                    h.update(chunk)
            if part.stat().st_size != size or (digest and digest != h.hexdigest()):
                raise ValueError(f'Incomplete or incorrect download of {name}')
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
    return path


def load_model(scale, weights_dir, device='cpu'):
    """Construct the released architecture and load all parameters strictly."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    name = 'simplefold_' + scale
    checkpoint = resource(weights_dir, name + '.ckpt')
    config = files('simplefold.configs').joinpath(f'model/architecture/foldingdit_{scale}.yaml')
    model = instantiate(OmegaConf.load(str(config)))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True), strict=True)
    return model.to(device).eval()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights_dir', type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    for filename in [f'simplefold_{s}.ckpt' for s in SCALES] + [ESM + '.pt', ESM + '-contact-regression.pt', 'ccd.pkl']:
        print(resource(args.weights_dir, filename, verify=args.verify), flush=True)
