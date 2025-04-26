import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from NextLayerPred import MultiLayerPredictor
from torch.utils.data import DataLoader
from CharacterLoader import CharacterLayerLoader

def evaluate_model(model, dataloader, num_timing_runs=100):
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
    layer_tensor, _ = next(iter(dataloader))
    layer_tensor = layer_tensor.to(device)

    # Split layers
    layer1 = layer_tensor[:, 0]   # Base layer
    gt_layer2 = layer_tensor[:, 1]
    gt_layer3 = layer_tensor[:, 2]
    gt_layer4 = layer_tensor[:, 3]
    gt_layer5 = layer_tensor[:, 4]

    # Calculate FLOPs using actual data
    pred_layers = model(
        layer1=layer1, 
        gt_layer2=gt_layer2,
        gt_layer3=gt_layer3,
        gt_layer4=gt_layer4,
        gt_layer5=gt_layer5,
        teacher_forcing=False
    )

    # Calculate FLOPs
    flops = FlopCountAnalysis(model, (layer1, gt_layer2, gt_layer3, gt_layer4, gt_layer5))
    total_flops = flops.total()
    flops_table = flop_count_table(flops)

    # Calculate average inference time
    with torch.no_grad():
        times = []
        for _ in range(num_timing_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            _ = model(layer1, gt_layer2, gt_layer3, gt_layer4, gt_layer5)
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


if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loading
    data_folder = "../data"
    dataset = CharacterLayerLoader(data_folder=data_folder, resolution=(100, 100))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, pin_memory=True)

    # Load and evaluate ViT model
    vit_model = MultiLayerPredictor(       
        image_size=100,
        patch_size=10,
        in_channels=3,
        embed_dim=256,
        nhead=4,
        num_layers=4
    ).to(device)
    vit_model.load_state_dict(torch.load('../models/vit_model.pth'))
    
    # Evaluate the model
    results = evaluate_model(vit_model, dataloader)
    print_evaluation_results(results)