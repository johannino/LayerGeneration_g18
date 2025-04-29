import os
import argparse
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from tqdm import tqdm
from NextLayerPred import MultiLayerPredictor
from diffusers import UNet2DConditionModel
from CharacterLoader import CharacterLayerLoader

# for FID (install via `pip install pytorch-fid`)
from pytorch_fid import fid_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model',      choices=['ViT','UNet'], required=True)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--device',     type=str, default='cuda')
    p.add_argument('--sample',     action='store_true',
                   help='Whether to draw+save samples')
    p.add_argument('--num_samples',type=int, default=256,
                   help='How many examples to sample for FID')
    p.add_argument('--real_dir',   type=str, default='real_samples',
                   help='Where to dump real images')
    p.add_argument('--gen_dir',    type=str, default='gen_samples',
                   help='Where to dump generated images')
    p.add_argument('--fid_batch',  type=int, default=50,
                   help='Batch size for FID computation')
    p.add_argument('--fid_dims',   type=int, default=2048,
                   help='Inception dims for FID')
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
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = model(*inputs)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    return total_flops, table, float(np.mean(times))

def sample_and_save(model, model_name, dataloader, num_samples, real_dir, gen_dir, device):
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir,  exist_ok=True)
    to_img = T.ToPILImage()

    count = 0
    model.eval().to(device)
    with torch.no_grad():
        for layers, _ in tqdm(dataloader, desc=f"Sampling {model_name}"):
            B = layers.size(0)
            layers = layers.to(device)
            base = layers[:,0]   # layer1
            # depending on model, roll forward:
            if model_name=='ViT':
                # we assume forward returns a tuple of 5 preds
                preds = model(
                    layer1=base,
                    gt_layer2=layers[:,1],
                    gt_layer3=layers[:,2],
                    gt_layer4=layers[:,3],
                    gt_layer5=layers[:,4],
                    teacher_forcing=False
                )
                final = preds[-1]
            else:  # UNet
                cond = torch.zeros((B,1,128), device=device)
                cur  = base
                # 5 iterative U-Net calls:
                for _ in range(5):
                    out = model(sample=cur,
                                timestep=torch.zeros(B,device=device,dtype=torch.long),
                                encoder_hidden_states=cond
                               ).sample
                    cur = out
                final = cur

            for i in range(B):
                if count>=num_samples: return
                # save real final layer
                real_img = to_img((layers[i,-1].cpu().clamp(0,1)))
                real_img.save(os.path.join(real_dir,  f"{count:04d}.png"))
                # save generated
                gen_img  = to_img((final[i].cpu().clamp(0,1)))
                gen_img.save(os.path.join(gen_dir,   f"{count:04d}.png"))
                count += 1

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # build dataset (only need the final layer for FID/ref)
    ds = CharacterLayerLoader(data_folder="../data_layers", resolution=(100,100))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # load model
    if args.model=='ViT':
        model = MultiLayerPredictor(
            image_size=100, patch_size=10,
            in_channels=3, embed_dim=256,
            nhead=4, num_layers=4
        )
        state = torch.load('../models/vit_model.pth', map_location='cpu')
        model.load_state_dict(state)
        # prepare FLOP/time inputs
        dummy = torch.randn(1,3,100,100,device=device)
        inputs = (
            dummy,
            torch.randn(1,3,100,100,device=device),
            torch.randn(1,3,100,100,device=device),
            torch.randn(1,3,100,100,device=device),
            torch.randn(1,3,100,100,device=device),
        )
    else:
        from diffusers import UNet2DConditionModel
        model = UNet2DConditionModel(
            sample_size=100, in_channels=3, out_channels=3,
            layers_per_block=2,
            block_out_channels=(64,128,128,256),
            down_block_types=("DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D"),
            up_block_types=("UpBlock2D","AttnUpBlock2D","UpBlock2D","UpBlock2D"),
            cross_attention_dim=128
        )
        state = torch.load('../models/unet_model.pth', map_location='cpu')
        model.load_state_dict(state)
        # dummy inputs for FLOP/time
        inputs = (
            torch.randn(1,3,100,100,device=device),          # sample
            torch.zeros(1,device=device,dtype=torch.long),   # timestep
            torch.zeros(1,1,128,device=device)               # cond
        )

    # FLOPs & timing
    flops, table, ms = flop_and_time(model, args.model, inputs, device)
    print(f"\n=== {args.model} Benchmark ===")
    print(f"Total FLOPs: {flops:,}")
    print(table)
    print(f"Avg latency (1-batch): {ms:.2f} ms")

    # optional sampling + FID
    if args.sample:
        sample_and_save(
            model, args.model, dl,
            args.num_samples, args.real_dir, args.gen_dir,
            device
        )
        print(f"\nSaved {args.num_samples} real→{args.real_dir} and gen→{args.gen_dir}")
        fid = fid_score.calculate_fid_given_paths(
            [args.real_dir, args.gen_dir],
            batch_size=args.fid_batch,
            device=str(device),
            dims=args.fid_dims
        )
        print(f"\nFID ({args.model}): {fid:.2f}")

if __name__=='__main__':
    main()
