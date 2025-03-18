import torch
import timm
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
import psutil
import os
import platform
import argparse
from torch.profiler import profile, record_function, ProfilerActivity
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Function to get current process memory usage in GB
def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # Convert to GB

# Models to benchmark
MODEL_NAMES = [
    # ViT variants
    "vit_base_patch16_224"
]
    # "vit_small_patch16_224", 
    # "vit_tiny_patch16_224",
#     # DeiT variants
#     "deit_tiny_patch16_224", 
#     "deit_small_patch16_224", 
#     "deit_base_patch16_224",
#     # Swin Transformer variants
#     "swin_tiny_patch4_window7_224", 
#     "swin_small_patch4_window7_224", 
#     "swin_base_patch4_window7_224",
#     # Hybrid ViT variants
#     "resnetv2_50x1_bit.goog_in21k_ft_in1k",
#     "coat_tiny",
#     "crossvit_tiny_240",
#     # EfficientFormer variants
#     "efficientformer_l1", 
#     "efficientformer_l3", 
#     "efficientformer_l7",
#     # MobileViT variants
#     "mobilevit_xxs", 
#     "mobilevit_xs", 
#     "mobilevit_s",
#     # MaxViT variants
#     "maxvit_tiny_tf_224", 
#     "maxvit_small_tf_224"
# ]

# Batch sizes to test
BATCH_SIZES = [1, 4, 8, 16, 32]

# Number of warmup and measurement iterations
WARMUP_ITERATIONS = 10
MEASUREMENT_ITERATIONS = 50

def load_images(image_dir, batch_size, max_images=1000):
    """
    Load images from a directory and prepare them for model input
    Returns a batch of preprocessed images
    """
    # Standard preprocessing for pretrained models
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(Path(image_dir).glob(f'*{ext}')))
        image_paths.extend(list(Path(image_dir).glob(f'*{ext.upper()}')))
    
    if not image_paths:
        raise ValueError(f"No images found in directory: {image_dir}")
    
    print(f"Found {len(image_paths)} images in {image_dir}")
    
    # Limit number of images to process
    image_paths = image_paths[:max_images]
    
    # Load and preprocess images
    images = []
    for img_path in tqdm(image_paths, desc="Loading images"):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = preprocess(img)
            images.append(img_tensor)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    if not images:
        raise ValueError("No valid images could be loaded")
    
    # Create batches
    batches = []
    for i in range(0, len(images), batch_size):
        if i + batch_size <= len(images):
            batch = torch.stack(images[i:i+batch_size])
            batches.append(batch)
        else:
            # For the last batch that might be smaller than batch_size
            # Just repeat the last image to make a full batch
            batch = images[i:]
            while len(batch) < batch_size:
                batch.append(images[0])  # Repeat first image to complete batch
            batch = torch.stack(batch)
            batches.append(batch)
    
    print(f"Created {len(batches)} batches of size {batch_size}")
    return batches

def benchmark_model(model_name, device, image_dir, batch_sizes=BATCH_SIZES):
    """
    Benchmark a single model across different batch sizes using real images
    Returns a dictionary with timing and memory results
    """
    results = []
    
    for batch_size in batch_sizes:
        print(f"Testing {model_name} with batch size {batch_size} on {device}")
        
        try:
            # Load images
            image_batches = load_images(image_dir, batch_size)
            
            # Load model
            model = timm.create_model(model_name, pretrained=True)
            model = model.to(device)
            model.eval()
            
            if device == "cuda":
                # Move batches to GPU
                image_batches = [batch.to(device) for batch in image_batches]
                # Clear GPU cache
                torch.cuda.empty_cache()
                # Record starting GPU memory
                torch.cuda.reset_peak_memory_stats()
                start_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB
            
            # Record starting CPU memory
            start_cpu_memory = get_process_memory()
            
            # Warmup runs (use first batch for warmup)
            for _ in range(WARMUP_ITERATIONS):
                with torch.no_grad():
                    _ = model(image_batches[0])
            
            # Synchronize if using GPU
            if device == "cuda":
                torch.cuda.synchronize()
            
            # Measure inference time
            latencies = []
            num_batches = min(len(image_batches), MEASUREMENT_ITERATIONS)
            
            # Use first batch for remaining measurements if not enough batches
            if num_batches < MEASUREMENT_ITERATIONS:
                print(f"Using first batch for remaining measurements (have {num_batches}, need {MEASUREMENT_ITERATIONS})")
            
            for i in range(MEASUREMENT_ITERATIONS):
                # Cycle through available batches
                batch_idx = i % num_batches
                input_tensor = image_batches[batch_idx]
                
                start_time = time.time()
                with torch.no_grad():
                    _ = model(input_tensor)
                
                # Synchronize if using GPU
                if device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate average latency and std deviation
            avg_latency = np.mean(latencies)
            std_latency = np.std(latencies)
            
            # Measure memory usage
            if device == "cuda":
                gpu_memory_used = torch.cuda.max_memory_allocated() / (1024 ** 3) - start_gpu_memory  # GB
            else:
                gpu_memory_used = 0
            
            cpu_memory_used = get_process_memory() - start_cpu_memory  # GB
            
            # Get model parameters count
            param_count = sum(p.numel() for p in model.parameters()) / 1e6  # Millions
            
            results.append({
                "model": model_name,
                "batch_size": batch_size,
                "latency_ms": avg_latency,
                "latency_std_ms": std_latency,
                "gpu_memory_gb": gpu_memory_used,
                "cpu_memory_gb": cpu_memory_used,
                "params_millions": param_count,
                "device": device
            })
            
            # Clean up
            del model, image_batches
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error benchmarking {model_name} with batch size {batch_size}: {str(e)}")
            results.append({
                "model": model_name,
                "batch_size": batch_size,
                "latency_ms": float('nan'),
                "latency_std_ms": float('nan'),
                "gpu_memory_gb": float('nan'),
                "cpu_memory_gb": float('nan'),
                "params_millions": float('nan'),
                "device": device,
                "error": str(e)
            })
    
    return results

def run_all_benchmarks(image_dir, device_list=["cpu", "cuda"]):
    """
    Run benchmarks for all models on specified devices
    """
    all_results = []
    
    for device in device_list:
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, skipping GPU benchmarks")
            continue
            
        for model_name in tqdm(MODEL_NAMES, desc=f"Benchmarking on {device}"):
            results = benchmark_model(model_name, device, image_dir)
            all_results.extend(results)
            
    return pd.DataFrame(all_results)

def plot_results(df, metric="latency_ms", batch_size=1):
    """
    Plot results for a specific metric and batch size
    """
    subset = df[df["batch_size"] == batch_size].copy()
    subset = subset.sort_values(metric)
    
    plt.figure(figsize=(12, 8))
    
    # Separate by device
    for device in subset["device"].unique():
        device_data = subset[subset["device"] == device]
        plt.barh(device_data["model"], device_data[metric], label=device)
    
    plt.xlabel(metric)
    plt.ylabel("Model")
    plt.title(f"{metric} for Batch Size {batch_size}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric}_bs{batch_size}.png")
    plt.close()

def visualize_all_results(df):
    """
    Create comprehensive visualizations of the results
    """
    # Create a directory for plots
    os.makedirs("plots", exist_ok=True)
    
    # Plot latency vs batch size for each model on CPU
    plt.figure(figsize=(14, 10))
    cpu_data = df[df["device"] == "cpu"]
    
    for model in MODEL_NAMES:
        model_data = cpu_data[cpu_data["model"] == model]
        if not model_data.empty:
            plt.plot(model_data["batch_size"], model_data["latency_ms"], marker='o', label=model)
    
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title("CPU Latency vs Batch Size")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/cpu_latency_vs_batchsize.png")
    plt.close()
    
    # If GPU data exists, plot GPU latency vs batch size
    if "cuda" in df["device"].unique():
        plt.figure(figsize=(14, 10))
        gpu_data = df[df["device"] == "cuda"]
        
        for model in MODEL_NAMES:
            model_data = gpu_data[gpu_data["model"] == model]
            if not model_data.empty:
                plt.plot(model_data["batch_size"], model_data["latency_ms"], marker='o', label=model)
        
        plt.xlabel("Batch Size")
        plt.ylabel("Latency (ms)")
        plt.title("GPU Latency vs Batch Size")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/gpu_latency_vs_batchsize.png")
        plt.close()
    
    # Plot parameters vs latency for batch size 1
    bs1_data = df[df["batch_size"] == 1]
    plt.figure(figsize=(14, 10))
    
    for device in bs1_data["device"].unique():
        device_data = bs1_data[bs1_data["device"] == device]
        plt.scatter(device_data["params_millions"], device_data["latency_ms"], 
                   label=device, alpha=0.7, s=100)
        
        # Add model names as annotations
        for _, row in device_data.iterrows():
            plt.annotate(row["model"].split('_')[0], 
                        (row["params_millions"], row["latency_ms"]),
                        fontsize=8)
    
    plt.xlabel("Parameters (Millions)")
    plt.ylabel("Latency (ms)")
    plt.title("Parameters vs Latency (Batch Size 1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/params_vs_latency.png")
    plt.close()

def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Benchmark vision transformer models with real images")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing images for benchmarking")
    parser.add_argument("--cpu_only", action="store_true", 
                        help="Run benchmarks only on CPU")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=BATCH_SIZES,
                        help="Batch sizes to benchmark (default: 1, 4, 8, 16, 32)")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific models to benchmark (default: all models)")
    
    return parser.parse_args()

def run_and_save_benchmarks(args):
    """
    Run all benchmarks and save results
    """
    # Update batch sizes if provided
    global BATCH_SIZES
    if args.batch_sizes:
        BATCH_SIZES = args.batch_sizes
        print(f"Using custom batch sizes: {BATCH_SIZES}")
    
    # Update model list if provided
    global MODEL_NAMES
    if args.models:
        MODEL_NAMES = args.models
        print(f"Using custom model list: {MODEL_NAMES}")
    
    # Check if CUDA is available
    device_list = ["cpu"]
    if not args.cpu_only and torch.cuda.is_available():
        device_list.append("cuda")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        if args.cpu_only:
            print("Running CPU benchmarks only as requested")
        else:
            print("CUDA not available, running CPU benchmarks only")
    
    # Get system info
    system_info = {
        "cpu": platform.processor(),
        "ram": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "image_dir": args.image_dir
    }
    
    print(f"System info: {system_info}")
    
    # Run benchmarks
    results_df = run_all_benchmarks(args.image_dir, device_list)
    
    # Save raw results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_df.to_csv(f"vit_benchmark_results_{timestamp}.csv", index=False)
    
    # Create output directory for JSON results
    os.makedirs("results", exist_ok=True)
    
    # Save system info
    with open(f"results/system_info_{timestamp}.json", "w") as f:
        import json
        json.dump(system_info, f, indent=2)
    
    # Generate basic plots
    for batch_size in BATCH_SIZES:
        plot_results(results_df, "latency_ms", batch_size)
        if "cuda" in device_list:
            plot_results(results_df, "gpu_memory_gb", batch_size)
        plot_results(results_df, "cpu_memory_gb", batch_size)
    
    # Generate comprehensive visualizations
    visualize_all_results(results_df)
    
    # Print summary for batch size 1
    print("\nSummary for batch size 1:")
    summary = results_df[results_df["batch_size"] == 1].sort_values("latency_ms")
    print(summary[["model", "device", "latency_ms", "gpu_memory_gb", "cpu_memory_gb", "params_millions"]])
    
    return results_df, system_info

if __name__ == "__main__":
    args = parse_arguments()
    results_df, system_info = run_and_save_benchmarks(args)