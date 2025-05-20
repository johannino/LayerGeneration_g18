import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
import torchvision.transforms.functional as TF
from NextLayerPred import MultiLayerPredictor
from torch.utils.data import DataLoader
from CharacterLoader import CharacterLayerLoader
from MulanDataset import MulanLayerDataset
from diffusers import UNet2DModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import os
import shutil
from pytorch_fid import fid_score

def evaluate_model(model, dataloader, model_name, num_timing_runs=100):
    """
    Evaluate a model's FLOPs and inference time.
    
    Args:
        model: The PyTorch model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        num_timing_runs: Number of runs for timing average
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Get a batch of data
    layer_tensor,_ = next(iter(dataloader))
    layer_tensor = layer_tensor.to(device)

    # Split layers
    layer1 = layer_tensor[:, 0]   # Base layer
    gt_layer2 = layer_tensor[:, 1]
    gt_layer3 = layer_tensor[:, 2]
    gt_layer4 = layer_tensor[:, 3]
    gt_layer5 = layer_tensor[:, 4]

    match model_name:
        case "ViT":
            pred_layers = model(
                layer1=layer1, 
                gt_layer2=gt_layer2,
                gt_layer3=gt_layer3,
                gt_layer4=gt_layer4,
                gt_layer5=gt_layer5,
                teacher_forcing=False
            )
            flops = FlopCountAnalysis(model, (layer1, gt_layer2, gt_layer3, gt_layer4, gt_layer5))
            total_flops = flops.total()
            flops_table = flop_count_table(flops)

        case "UNet":
            
            num_inference_steps = 5
            batch_size = layer1.shape[0]
            
            # Initial input is layer1
            current_layer = layer1
            generated_layers = [current_layer.squeeze(0).cpu()] 

            for _ in range(5):
                predicted_residual = model(
                    sample=current_layer,
                    timestep=torch.zeros(1, device=device, dtype=torch.long),
                ).sample
                
                next_layer = current_layer + predicted_residual
                generated_layers.append(next_layer.squeeze(0).cpu())
                current_layer = next_layer

            dummy_sample = layer1 
            dummy_timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
            
            flops = FlopCountAnalysis(
                model, 
                inputs=(
                    dummy_sample,
                    dummy_timestep,
                )
            )
            flops_per_step = flops.total() 
            total_flops = flops_per_step * num_inference_steps 
            flops_table = flop_count_table(flops)

        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    with torch.no_grad():
        times = []
        for _ in range(num_timing_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            match model_name:
                case "ViT":
                    _ = model(layer1, gt_layer2, gt_layer3, gt_layer4, gt_layer5)
                case "UNet":
                    current_layer = layer1
                    for _ in range(5):
                        pred_residual = model(
                            sample=layer1,
                            timestep=torch.zeros(batch_size, device=device, dtype=torch.long),
                        ).sample
                        current_layer = pred_residual

            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        avg_time = np.mean(times)

    return {
        'total_flops': total_flops,
        'flops_table': flops_table,
        'avg_inference_time': avg_time
    }

def calculate_predictions_per_second(avg_inference_time_ms, batch_size):
    """
    Calculate the number of predictions per second.
    
    Args:
        avg_inference_time_ms: Average inference time in milliseconds
        batch_size: Number of samples in each batch
    
    Returns:
        float: Number of predictions per second
    """
    seconds_per_inference = avg_inference_time_ms / 1000  # Convert ms to seconds
    predictions_per_second = batch_size / seconds_per_inference
    return predictions_per_second

def print_evaluation_results(results):
    """Print the evaluation results in a formatted way"""
    print("\nFLOPs Analysis:")
    print(f"Total FLOPs: {results['total_flops']:.2e}")
    print("\nDetailed FLOPs breakdown:")
    print(results['flops_table'])
    print(f"\nAverage inference time: {results['avg_inference_time']:.2f} ms")

    batch_size = 32
    pps = calculate_predictions_per_second(results['avg_inference_time'], batch_size)
    print(f"Predictions per second: {pps:.2f}")
    print()

def plot_generated_samples(model, dataloader, model_name, num_samples=250):
    """
    Plot generated samples from the model.
    
    Args:
        model: The PyTorch model to evaluate
        dataloader: DataLoader containing the evaluation data
        num_samples: Number of samples to plot
    """
    print("Plotting samples for", model_name)
    os.makedirs(f'../figures/generated_samples/{model_name}', exist_ok=True)
    os.makedirs(f'../figures/generated_samples/{model_name}/final_layer', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        for i, (layer_tensor,_) in enumerate(dataloader):
            if i >= num_samples:
                break
            
            layer_tensor = layer_tensor.to(device)
            layer1 = layer_tensor[:, 0]
            gt_layer2 = layer_tensor[:, 1]
            gt_layer3 = layer_tensor[:, 2]
            gt_layer4 = layer_tensor[:, 3]
            gt_layer5 = layer_tensor[:, 4]

            match model_name:
                case "ViT":
                    pred_layers = model(
                        layer1=layer1, 
                        gt_layer2=gt_layer2,
                        gt_layer3=gt_layer3,
                        gt_layer4=gt_layer4,
                        gt_layer5=gt_layer5,
                        teacher_forcing=False
                    )
                    pred_layers = [layer1.squeeze(0).cpu()] + [layer.squeeze(0).cpu() for layer in pred_layers]
                case "UNet":
                    
                    #initial_noise = torch.randn(1, 3, 256, 256, device=device)

                    #current_layer = model(
                    #    sample=initial_noise,
                    #    timestep=torch.zeros(1, device=device, dtype=torch.long),
                    #).sample
                    pred_layers = [layer1.squeeze(0).cpu()]

                    for _ in range(1, 6):
                        current_layer = pred_layers[-1].unsqueeze(0).to(device)
                        with torch.no_grad():
                            predicted_residual = model(
                                sample=current_layer,
                                timestep=torch.zeros(1, device=device, dtype=torch.long),
                            ).sample
                            
                            next_layer = current_layer + predicted_residual
                            pred_layers.append(next_layer.squeeze(0).cpu())
                            
                            current_layer = next_layer
            
            fig, axs = plt.subplots(1, 6, figsize=(20, 5))
            fig.suptitle(f"Generated Layers {model_name}", fontsize=32)
            for k, layer in enumerate(pred_layers):
                generated = layer.permute(1, 2, 0).clamp(0, 1)
                if k == 0:
                    title = "Input Layer"
                else:
                    title = f"Generated Layer {k}"
                axs[k].imshow(generated)
                axs[k].axis('off')
                axs[k].set_title(f"{title}", fontsize=16)
                if k == len(pred_layers) - 1:
                    plt.savefig(f'../figures/generated_samples/{model_name}/generated_layers_{model_name}_{i}.png')
                    plt.imsave(f'../figures/generated_samples/{model_name}/final_layer/final_layer_only_{model_name}_{i}.png', generated.cpu().numpy())
                    plt.close()

def calculate_fid(real_dir = '../data/face', gen_dir_last = '../figures/generated_samples/', model_name=None, keep_samples=10):
    """
    Calculate FID score between real and generated images, keeping only specified number of samples.
    
    Args:
        real_dir: Directory containing real images
        gen_dir_last: Base directory for generated images
        model_name: Name of the model ('ViT' or 'UNet')
        keep_samples: Number of samples to keep (default: 10)
    """
    # Setup directories
    gen_dir_not_last = gen_dir_last + model_name
    gen_dir_last = gen_dir_last + model_name + '/final_layer'
    
    # Get list of all generated images from final layer
    generated_files = [f for f in os.listdir(gen_dir_last) if f.endswith('.png')]
    generated_files_not_last = [f for f in os.listdir(gen_dir_not_last) if f.endswith('.png') and not os.path.isdir(os.path.join(gen_dir_not_last, f))]
    
    # Calculate FID using remaining files
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, gen_dir_last],
        batch_size=50,
        device=str(device),
        dims=2048,
    )

    # Select random samples to keep
    files_to_keep = set(random.sample(generated_files, min(keep_samples, len(generated_files))))
    files_to_keep_not_last = set(random.sample(generated_files_not_last, min(keep_samples, len(generated_files_not_last))))
    
    # Delete files in main directory (excluding final_layer directory)
    for file in generated_files_not_last:
        file_path = os.path.join(gen_dir_not_last, file)
        if os.path.isfile(file_path) and file not in files_to_keep_not_last:
            os.remove(file_path)
    
    # Delete files in final_layer directory
    for file in generated_files:
        if file not in files_to_keep:
            os.remove(os.path.join(gen_dir_last, file))
    
    print(f"\nFID: {fid:.2f} for {model_name} (kept {len(files_to_keep)} samples)")
    
    return fid


if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    data_folder = "../data"
    dataset = CharacterLayerLoader(data_folder=data_folder, resolution=(256, 256))
    #dataset = MulanLayerDataset('../MULAN_data')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

    # Load and evaluate ViT model
    vit_model = MultiLayerPredictor(       
        image_size=256,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        nhead=4,
        num_layers=4
    ).to(device)
    vit_model.load_state_dict(torch.load('../models/vit_model.pth'))
    
    # Evaluate the model
    vit_results = evaluate_model(vit_model, dataloader, "ViT")
    print_evaluation_results(vit_results)

    print("Evaluating Unet model...")

    unet_model = UNet2DModel(
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels= (64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        mid_block_type="AttnMidBlock2D",
    ).to(device)

    unet_model.load_state_dict(torch.load('../models/unet_model.pth'))
    unet_results = evaluate_model(unet_model, dataloader, "UNet")
    print_evaluation_results(unet_results)

    samples = 2000
    plot_generated_samples(unet_model, dataloader, "UNet",num_samples=samples)
    plot_generated_samples(vit_model, dataloader, "ViT",num_samples=samples)
    calculate_fid(model_name='ViT')
    calculate_fid(model_name='UNet')

    print("\nModel Comparison:")
    print(f"ViT vs UNet:")
    print(f"FLOPs ratio: {vit_results['total_flops'] / unet_results['total_flops']:.2f}x")
    print(f"Inference time ratio: {vit_results['avg_inference_time'] / unet_results['avg_inference_time']:.2f}x")
