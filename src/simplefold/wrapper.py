#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import torch
from pathlib import Path

from simplefold.conditioning import Conditioner, load_esm
from simplefold.inference import (
    initialize_esm_model, initialize_folding_model, initialize_others, initialize_plddt_module,
)

from simplefold.utils.datamodule_utils import process_one_inference_structure
from simplefold.utils.boltz_utils import process_structure, save_structure
from simplefold.utils.fasta_utils import process_fastas, download_fasta_utilities

try: 
    import mlx.core as mx
    MLX_AVAILABLE = True
except:
    MLX_AVAILABLE = False
    print("MLX not installed, skip importing MLX related packages.")


class ModelWrapper:
    def __init__(
        self,
        simplefold_model,
        plddt=False,
        ckpt_dir="./artifacts",
        backend="torch",
    ):
        self.simplefold_model = simplefold_model
        self.plddt = plddt
        self.ckpt_dir = Path(ckpt_dir)
        self.backend = backend
        if self.backend == "mlx" and not MLX_AVAILABLE:
            self.backend = "torch"
            print("MLX not installed, switch to torch backend.")
        self.folding_model = None
        self.plddt_out_module = None
        self.plddt_latent_module = None
        self.device = self.get_device()
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def get_device(self):
        if self.backend == "torch":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif self.backend == "mlx":
            device = "cpu"
        return device

    def from_pretrained_folding_model(self):
        model, _ = initialize_folding_model(self)
        return model

    def from_pretrained_plddt_model(self):
        latent, output = initialize_plddt_module(self, self.device)
        return {'plddt_out_module': output, 'plddt_latent_module': latent}


class InferenceWrapper:
    def __init__(
        self,
        output_dir,
        prediction_dir,
        num_steps,
        nsample_per_protein,
        tau,
        device,
        backend,
        cache_dir=None,
    ):
        self.num_steps = num_steps
        self.nsample_per_protein = nsample_per_protein
        self.tau = tau
        self.device = device
        self.backend = backend

        if self.backend == "mlx" and not MLX_AVAILABLE:
            self.backend = "torch"
            print("MLX not installed, switch to torch backend.")

        # create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Keep reusable reference data separate from per-run intermediate files.
        cache = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache/simplefold"
        cache.mkdir(parents=True, exist_ok=True)
        work_dir = output_dir / "cache"
        work_dir.mkdir(parents=True, exist_ok=True)

        # create prediction directory
        prediction_dir = output_dir / prediction_dir
        prediction_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = output_dir
        self.cache = cache
        self.work_dir = work_dir
        self.prediction_dir = prediction_dir

        self.conditioner = Conditioner(cache, device) if self.backend == 'torch' else None
        self.esm_model = self.esm_dict = self.af2_to_esm = None
        if self.conditioner is None:
            self.initialize_esm_model()
        self.initialize_others()

    def initialize_esm_model(self):
        if self.conditioner is None:
            parts = initialize_esm_model(self, self.device)
        else:
            parts = load_esm(self.conditioner.weights_dir, self.device)
        self.esm_model, self.esm_dict, self.af2_to_esm = parts

    def initialize_others(self):
        self.tokenizer, self.featurizer, self.processor, self.flow, self.sampler = initialize_others(self, self.device)

    def process_input(self, aa_seq):
        # process fasta files to input format
        ccd_dir = self.conditioner.weights_dir if self.conditioner else self.cache
        download_fasta_utilities(ccd_dir)
        # save the input sequence to a fasta file
        with open(self.work_dir / "input.fasta", "w") as f:
            f.write(f">A|Protein\n{aa_seq}\n")
        data = [self.work_dir / "input.fasta"]
        process_fastas(
            data=data,
            out_dir=self.work_dir,
            ccd_path=ccd_dir / "ccd.pkl",
        )

        # prepare the target protein data for inference
        struct_file = self.work_dir / "structures" / "input.npz"
        record_file = self.work_dir / "records" / "input.json"
        batch, structure, record = process_one_inference_structure(
            struct_file,
            record_file,
            self.tokenizer,
            self.featurizer,
            self.processor,
            self.esm_model,
            self.esm_dict,
            self.af2_to_esm,
        )
        if self.conditioner is not None:
            self.conditioner.add_embedding(batch)
        return batch, structure, record

    def run_inference(self, batch, model, plddt_model, device):
        # run inference for target protein
        if self.backend == "torch":
            noise = torch.randn_like(batch["coords"]).to(device)
        elif self.backend == "mlx":
            noise = mx.random.normal(batch["coords"].shape)
        out_dict = self.sampler.sample(model, self.flow, noise, batch)

        plddt_out_module = plddt_model["plddt_out_module"]
        plddt_latent_module = plddt_model["plddt_latent_module"]

        if plddt_latent_module is None or plddt_out_module is None:
            plddts = None
        else:
            if self.backend == "torch":
                t = torch.ones(batch["coords"].shape[0], device=device)
                # use unscaled coords to extract latent for pLDDT prediction
                out_feat = plddt_latent_module(
                    out_dict["denoised_coords"].detach(), t, batch
                )
                plddt_out_dict = plddt_out_module(
                    out_feat["latent"].detach(),
                    batch,
                )
            elif self.backend == "mlx":
                t = mx.ones(batch["coords"].shape[0])
                # use unscaled coords to extract latent for pLDDT prediction
                out_feat = plddt_latent_module(out_dict["denoised_coords"], t, batch)
                plddt_out_dict = plddt_out_module(
                    out_feat["latent"],
                    batch,
                )
            # scale pLDDT to [0, 100]
            plddts = plddt_out_dict["plddt"] * 100.0

        out_dict = self.processor.postprocess(out_dict, batch)
        # sampled_coord = out_dict['denoised_coords'].detach()
        if self.backend == "torch":
            sampled_coord = out_dict["denoised_coords"].detach()
        else:
            sampled_coord = out_dict["denoised_coords"]

        return {
            "sampled_coord": sampled_coord,
            "pad_mask": batch["atom_pad_mask"],
            "plddts": plddts,
        }

    def save_result(self, structure, record, results, out_name):
        sampled_coord = results["sampled_coord"]
        pad_mask = results["pad_mask"]
        plddt = results["plddts"]

        save_paths = []
        for i in range(sampled_coord.shape[0]):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]
            plddt_i = plddt[i] if plddt is not None else None
            out_name_i = f"{out_name}_sampled_{i}"
            # save the generated structure
            structure_save = process_structure(
                structure, sampled_coord_i, pad_mask_i, record, backend=self.backend
            )
            save_structure(
                structure_save,
                self.prediction_dir,
                out_name_i,
                output_format="mmcif",
                plddts=plddt_i,
            )
            save_paths.append(self.prediction_dir / f"{out_name_i}.cif")
        return save_paths
