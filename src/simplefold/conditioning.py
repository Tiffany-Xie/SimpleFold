"""Sequence-only Boltz features and reusable, full-precision ESM-2 conditioning."""

import hashlib
import pickle
from dataclasses import asdict
from pathlib import Path

import torch
from esm.pretrained import load_model_and_alphabet_core

from simplefold.assets import ESM, RELEASES, cache_lock, resource
from simplefold.boltz_data_pipeline import const
from simplefold.boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from simplefold.boltz_data_pipeline.parse.schema import parse_boltz_schema
from simplefold.boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from simplefold.processor.protein_processor import ProteinDataProcessor
from simplefold.utils.datamodule_utils import featurize_structure
from simplefold.utils.esm_utils import _af2_to_esm


def load_esm(weights_dir, device):
    # Meta's checkpoint contains its configuration as well as the state_dict.
    # The two release downloads have pinned checksums in assets.py.
    weights = torch.load(resource(weights_dir, ESM + '.pt'), map_location='cpu', weights_only=False)
    regression = torch.load(resource(weights_dir, ESM + '-contact-regression.pt'),
                            map_location='cpu', weights_only=True)
    model, alphabet = load_model_and_alphabet_core(ESM, weights, regression)
    return model.eval().to(device), alphabet, _af2_to_esm(alphabet).to(device)


class Conditioner:
    def __init__(self, root, device='cpu', *, weights_dir=None):
        self.root = Path(root)
        self.weights_dir = Path(weights_dir) if weights_dir is not None else self.root / 'weights'
        self.device = torch.device(device)
        self.processor = ProteinDataProcessor(self.device)
        self.tokenizer = BoltzTokenizer()
        self.featurizer = BoltzFeaturizer()
        self.ccd = None
        self.esm = None

    @torch.no_grad()
    def batch(self, sequence):
        """Featurize a single canonical chain from sequence, independently of native XYZ."""
        if not sequence or set(sequence) - set('ACDEFGHIKLMNPQRSTVWY'):
            raise ValueError('Expected one nonempty canonical protein sequence')
        if self.ccd is None:
            with resource(self.weights_dir, 'ccd.pkl').open('rb') as handle:
                ccd = pickle.load(handle)
            self.ccd = {key: ccd[key] for key in const.ref_atoms if key in ccd}
            del ccd
        schema = {'version': 1, 'sequences': [{'protein': {'id': 'A', 'sequence': sequence,
                  'modifications': [], 'msa': None}}], 'bonds': []}
        target = parse_boltz_schema('sequence', schema, self.ccd)
        batch = featurize_structure(target.structure, asdict(target.record), self.tokenizer, self.featurizer)
        if batch['aa_seq'] != [sequence]:
            raise ValueError('Tokenizer changed the input sequence')
        batch = self.processor.preprocess_inference(batch)
        self.add_embedding(batch)
        return batch

    @torch.no_grad()
    def add_embedding(self, batch):
        """Populate ESM features for one target, including repeated inference copies."""
        sequence, = batch['aa_seq']
        length = int(batch['cropped_num_tokens'][0])
        identity = f'{ESM}/{RELEASES[ESM + ".pt"][1]}/layers0-36-fp32-v1/{sequence}'
        path = self.root / 'embeddings' / (hashlib.sha256(identity.encode()).hexdigest() + '.pt')
        if not path.is_file():
            with cache_lock(path):
                if not path.is_file():
                    if self.esm is None:
                        self.esm = load_esm(self.weights_dir, self.device)
                    # Compute the 37 layers once, before checkpoint-specific layer mixing.
                    self.processor.process_esm(batch, *self.esm, inference=True)
                    embedding = batch['esm_s'][0, :length].detach().to('cpu').contiguous()
                    part = path.with_suffix('.part')
                    try:
                        torch.save(embedding, part)
                        part.replace(path)
                    finally:
                        part.unlink(missing_ok=True)
        embedding = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
        if embedding.shape != (length, 37, 2560) or embedding.dtype != torch.float32:
            raise ValueError(f'Invalid ESM embedding at {path}')
        batch['esm_s'] = embedding.to(self.device).unsqueeze(0).expand(batch['coords'].shape[0], -1, -1, -1)
