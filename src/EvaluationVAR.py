import os, sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from NextLayerPred import MultiLayerPredictor
from diffusers import UNet2DConditionModel
from CharacterLoader import CharacterLayerLoader
from MULANLoader import MulanLayerDataset
from pytorch_fid import fid_score

# Comment out the following lines if you don't have the VAR codebase
HERE = os.path.dirname(__file__)
VAR_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "VAR"))
sys.path.insert(0, VAR_ROOT)
from models import build_vae_var
from models.vqvae import VectorQuantizer2
from models.var import VAR
from models.helpers import sample_with_top_k_top_p_


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["ViT", "UNet", "VAR"], required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--sample", action="store_true", help="Whether to draw+save samples")
    p.add_argument("--dataset", type=str, default="Character", help="Dataset to use", choices=["Character", "MULAN"])
    p.add_argument(
        "--num_samples",
        type=int,
        default=256,
        help="How many examples to sample for FID",
    )
    p.add_argument(
        "--real_dir", type=str, default="real_samples", help="Where to dump real images"
    )
    p.add_argument(
        "--gen_dir_last",
        type=str,
        default="gen_samples",
        help="Where to dump generated images (last layer)",
    )
    p.add_argument(
        "--gen_dir_chain",
        type=str,
        default="gen_samples_chain",
        help="Where to dump generated images (full chain)",
    )
    p.add_argument(
        "--fid_batch", type=int, default=50, help="Batch size for FID computation"
    )
    p.add_argument("--fid_dims", type=int, default=2048, help="Inception dims for FID")
    p.add_argument(
        "--vae_ckpt",
        type=str,
        default="../models/vae_model.pth",
        help="Path to VAE checkpoint (VAR only)",
    )
    p.add_argument(
        "--var_ckpt",
        type=str,
        default="../models/var_model.pth",
        help="Path to VAR checkpoint (VAR only)",
    )
    return p.parse_args()


def flop_and_time(model, model_name, inputs, device):
    model.eval().to(device)
    # FLOPs
    flops = FlopCountAnalysis(model, inputs)
    total_flops = flops.total()
    table = flop_count_table(flops)

    # timing
    times = []
    with torch.no_grad():
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return total_flops, table, float(np.mean(times))


def sample_and_save(
    model,
    model_name,
    dataloader,
    num_samples,
    real_dir,
    gen_dir_last,
    gen_dir_chain,
    device,
):
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir_last, exist_ok=True)
    os.makedirs(gen_dir_chain, exist_ok=True)
    to_img = T.ToPILImage()

    count = 0
    model.eval().to(device)
    with torch.no_grad():
        for layers, _ in tqdm(dataloader, desc=f"Sampling {model_name}"):
            B = layers.size(0)
            layers = layers.to(device)
            base = layers[:, 0]  # input (layer1)

            # 1) generate either final-only or full chain
            if model_name == "ViT":
                # preds is tuple(pred2, pred3, pred4, pred5, pred6)
                preds = model(
                    layer1=base,
                    gt_layer2=layers[:, 1],
                    gt_layer3=layers[:, 2],
                    gt_layer4=layers[:, 3],
                    gt_layer5=layers[:, 4],
                    teacher_forcing=False,
                )
                chain = list(preds)  # length=5
                final = chain[-1]

            else:  # UNet
                cond = torch.zeros((B, 1, 128), device=device)
                cur = base
                chain = []
                for _ in range(5):
                    out = model(
                        sample=cur,
                        timestep=torch.zeros(B, device=device, dtype=torch.long),
                        encoder_hidden_states=cond,
                    ).sample
                    chain.append(out)
                    cur = out
                final = chain[-1]

            # 2) save images
            for i in range(B):
                if count >= num_samples:
                    return

                # real final-layer (for FID reference)
                real_img = to_img(layers[i, -1].cpu().clamp(0, 1))
                real_img.save(os.path.join(real_dir, f"{count:04d}.png"))

                # generated final-layer
                gen_last_img = to_img(final[i].cpu().clamp(0, 1))
                gen_last_img.save(os.path.join(gen_dir_last, f"{count:04d}.png"))

                # full chain montage: [base, pred2, ..., pred6]
                images = [base[i].cpu()] + [p[i].cpu() for p in chain]
                grid = torchvision.utils.make_grid(
                    images, nrow=len(images), pad_value=1.0
                )
                torchvision.utils.save_image(
                    grid,
                    os.path.join(gen_dir_chain, f"{count:04d}_chain.png"),
                    normalize=False,
                )

                count += 1


def sample_and_save_VAR_final(
    vae, var, dataloader, num_samples, real_dir, gen_dir_last, device, seed=0
):
    """
    For VAR we just call autoregressive_infer_cfg to get the final reconstructions,
    then save them alongside the real held-out final layers for FID.
    """
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir_last, exist_ok=True)

    vae.eval().to(device)
    var.eval().to(device)

    idx = 0
    B = None
    for batch in tqdm(dataloader, desc="Sampling VAR"):
        layers = batch
        if B is None:
            B = layers.size(0)
        if idx >= num_samples:
            break

        # move to device
        layers = layers.to(device)

        # save real final layer now (so we don't have to loop twice)
        final_real = layers[:, -1]  # [B,3,H,W]
        for i in range(final_real.size(0)):
            if idx >= num_samples:
                break
            torchvision.utils.save_image(
                final_real[i].cpu(),
                os.path.join(real_dir, f"{idx:06d}.png"),
                normalize=False,
            )
            idx += 1

        recon = var.autoregressive_infer_cfg(
            B=layers.size(0),
            label_B=torch.zeros(layers.size(0), dtype=torch.long, device=device),
            g_seed=seed + idx // B,
            cfg=1.0,
            top_k=0,
            top_p=0.0,
            more_smooth=False,
        )  # returns [B,3,H,W] in [0,1]

        # save generated final layers
        start = idx - final_real.size(0)
        for i in range(recon.size(0)):
            j = start + i
            if j < 0 or j >= num_samples:
                continue
            torchvision.utils.save_image(
                recon[i].cpu(),
                os.path.join(gen_dir_last, f"{j:06d}.png"),
                normalize=False,
            )

    print(f"Saved {min(idx, num_samples)} real→{real_dir}, gen→{gen_dir_last}")


def sample_and_save_VAR_chain(
    vae,
    var,
    patch_nums,
    dataloader,
    num_samples,
    real_dir,
    gen_dir_last,
    gen_dir_chain,
    device,
    seed=0,
):
    to_pil = T.ToPILImage()
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir_last, exist_ok=True)
    os.makedirs(gen_dir_chain, exist_ok=True)

    vae.eval().to(device)
    var.eval().to(device)

    idx = 0
    rng = torch.Generator(device=device)

    with torch.inference_mode():
        for layers, _ in tqdm(dataloader, desc="Sampling VAR chain"):
            B = layers.size(0)
            if idx >= num_samples:
                break

            # only take the final composed layer as the VAE input
            full_img = layers[:, -1].to(device)  # [B,3,H,W]

            # 1) encode → discrete indices → prepare VAR input
            gt_idx_list = vae.img_to_idxBl(full_img)  # list of [B, L_i]
            x_var_in = vae.quantize.idxBl_to_var_input(
                gt_idx_list
            )  # [B, L−first_l, Cvae]
            label_B = torch.zeros(
                B, dtype=torch.long, device=device
            )  # dummy unconditional

            # 2) get all per‐scale logits in one forward pass
            logits_list = var(label_B, x_var_in)  # list of [B, L_i, V]

            # 3) autoregressive build & decode chain
            f_hat = torch.zeros(
                B, vae.Cvae, patch_nums[-1], patch_nums[-1], device=device
            )
            chain_imgs = []
            for si, (logits_i, pn) in enumerate(zip(logits_list, patch_nums)):
                if logits_i.ndim == 2:
                    logits_i = logits_i.unsqueeze(1)  # ensure shape [B, L_i, V]

                rng.manual_seed(seed + idx)
                idxs = sample_with_top_k_top_p_(
                    logits_i, rng=rng, top_k=0, top_p=0.0, num_samples=1
                )[:, :, 0]  # [B, L_i]

                h = vae.quantize.embedding(idxs)  # [B, L_i, Cvae]
                h = h.transpose(1, 2).reshape(B, vae.Cvae, pn, pn)

                f_hat, _ = vae.quantize.get_next_autoregressive_input(
                    si, len(patch_nums), f_hat, h
                )

                img_i = vae.decoder(vae.post_quant_conv(f_hat)).clamp(-1, 1)
                img_i = (img_i + 1) * 0.5  # to [0,1]
                chain_imgs.append(img_i)

            # 4) save real, final, and full‐chain montage
            gen_final = chain_imgs[-1]
            for b in range(B):
                if idx >= num_samples:
                    break

                # real (ground truth)
                torchvision.utils.save_image(
                    full_img[b].cpu(),
                    os.path.join(real_dir, f"{idx:06d}.png"),
                    normalize=False,
                )

                # generated final-layer
                torchvision.utils.save_image(
                    gen_final[b].cpu(),
                    os.path.join(gen_dir_last, f"{idx:06d}.png"),
                    normalize=False,
                )

                # montage of every scale
                grid = torchvision.utils.make_grid(
                    [c[b].cpu() for c in chain_imgs],
                    nrow=len(chain_imgs),
                    pad_value=1.0,
                )
                torchvision.utils.save_image(
                    grid,
                    os.path.join(gen_dir_chain, f"{idx:06d}_chain.png"),
                    normalize=False,
                )

                idx += 1


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # build dataset (only need the final layer for FID/ref)
    if args.dataset == "MULAN":
        ds = MulanLayerDataset(data_folder="/work3/s243345/adlcv/project/MULAN_data", resolution=(256, 256))
    else:
        ds = CharacterLayerLoader(data_folder="../data", resolution=(256, 256))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # load model
    if args.model == "ViT" or args.model == "UNet":
        if args.model == "ViT":
            model = MultiLayerPredictor(
                image_size=256,
                patch_size=10,
                in_channels=3,
                embed_dim=256,
                nhead=4,
                num_layers=4,
            )
            state = torch.load("../models/vit_model.pth", map_location="cpu")
            model.load_state_dict(state)
            # prepare FLOP/time inputs
            dummy = torch.randn(1, 3, 100, 100, device=device)
            inputs = (
                dummy,
                torch.randn(1, 3, 100, 100, device=device),
                torch.randn(1, 3, 100, 100, device=device),
                torch.randn(1, 3, 100, 100, device=device),
                torch.randn(1, 3, 100, 100, device=device),
            )
        else:
            from diffusers import UNet2DConditionModel

            model = UNet2DConditionModel(
                sample_size=100,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(64, 128, 128, 256),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
                cross_attention_dim=128,
            )
            state = torch.load("../models/unet_model.pth", map_location="cpu")
            model.load_state_dict(state)
            # dummy inputs for FLOP/time
            inputs = (
                torch.randn(1, 3, 100, 100, device=device),  # sample
                torch.zeros(1, device=device, dtype=torch.long),  # timestep
                torch.zeros(1, 1, 128, device=device),  # cond
            )

        # FLOPs & timing
        flops, table, ms = flop_and_time(model, args.model, inputs, device)
        print(f"\n=== {args.model} Benchmark ===")
        print(f"Total FLOPs: {flops:,}")
        print(table)
        print(f"Avg latency (1-batch): {ms:.2f} ms")

        # Sampling + FID
        if args.sample:
            sample_and_save(
                model,
                args.model,
                dl,
                args.num_samples,
                args.real_dir,
                args.gen_dir_last,
                args.gen_dir_chain,
                device,
            )
            print(
                f"\nSaved {args.num_samples} real→{args.real_dir}, gen→{args.gen_dir_last}, {args.gen_dir_chain}"
            )
            fid = fid_score.calculate_fid_given_paths(
                [args.real_dir, args.gen_dir_last],
                batch_size=args.fid_batch,
                device=str(device),
                dims=args.fid_dims,
            )
            print(f"\nFID ({args.model}): {fid:.2f}")

    elif args.model == "VAR":
        # 1. build & load vae+var exactly as before
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        vae, var = build_vae_var(
            V=4096,
            Cvae=32,
            ch=160,
            share_quant_resi=4,
            device=device,
            patch_nums=patch_nums,
            num_classes=1,
            depth=16,
            shared_aln=False,
        )
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"), strict=True)
        ckpt = torch.load(args.var_ckpt, map_location="cpu")
        var.load_state_dict(ckpt["trainer"]["var_wo_ddp"], strict=True)
        vae.to(device).eval()
        var.to(device).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        for p in var.parameters():
            p.requires_grad_(False)

        # === FLOP counting for the exact sampling call ===
        class ARSampler(nn.Module):
            def __init__(self, var_model):
                super().__init__()
                self.var = var_model

            def forward(self, label_B):
                # note: we pick fixed values for all of the other args
                return self.var.autoregressive_infer_cfg(
                    B=label_B.shape[0],
                    label_B=label_B,
                    g_seed=0,
                    cfg=1.0, top_k=0, top_p=0.0, more_smooth=False
                )

        sampler = ARSampler(var).to(device)
        # create a dummy batch of labels
        dummy_labels = torch.zeros(args.batch_size, dtype=torch.long, device=device)
        # now fvcore can see through the module hierarchy
        flops = FlopCountAnalysis(sampler, (dummy_labels,))
        print(f"[VAR sampler] FLOPs per call: {flops.total():,}")
        print(flop_count_table(flops))

        # 2. make a dataloader (we only need the final layer to compare)
        if args.dataset == "MULAN":
            data_folder = "/work3/s243345/adlcv/project/MULAN_data"
            ds_var = MulanLayerDataset(data_folder=data_folder, resolution=(256, 256))
        else:
            data_folder = "../data"
            ds_var = CharacterLayerLoader(data_folder=data_folder, resolution=(256, 256))
        
        dl_var = DataLoader(
            ds_var, batch_size=args.batch_size, shuffle=False, pin_memory=True
        )

        # 3. sample & save

        sample_and_save_VAR_final(
            vae,
            var,
            dl_var,
            args.num_samples,
            args.real_dir,
            args.gen_dir_last,
            device,
            seed=0,
        )

        """
        sample_and_save_VAR_chain(
            vae, var, (1, 2, 3, 4, 5, 6, 8, 10, 13, 16), dl_var,
            args.num_samples, args.real_dir,
            args.gen_dir_last, args.gen_dir_chain,
            device, seed=0
        )
        
        # 3b. alternating scales ending with final
        selected_scales = [1, 3, 5, 10, 16]
        sample_and_save_VAR_chain(
            vae=vae,
            var=var,
            patch_nums=selected_scales,
            dataloader=dl_var,
            num_samples=args.num_samples,
            real_dir=args.real_dir,
            gen_dir_last=args.gen_dir_last,
            gen_dir_chain=args.gen_dir_chain,
            device=device,
            seed=0,
        )
        """
        
        # 4. compute FID
        fid = fid_score.calculate_fid_given_paths(
            [args.real_dir, args.gen_dir_last],
            batch_size=args.fid_batch,
            device=str(device),
            dims=args.fid_dims,
        )
        print(f"\nFID (VAR): {fid:.2f}")


if __name__ == "__main__":
    main()
