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
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Number of warmup and measurement iterations
WARMUP_ITERATIONS = 5
MEASUREMENT_ITERATIONS = 25

# Models to benchmark with their input sizes
# Format: [model_name, (image_size, image_size)]
MODELS = [
    ["efficientnet_b0", (224, 224)],
    ["vit_base_patch16_224", (224, 224)],
    ["vit_small_patch16_224", (224, 224)],
    ["resnet50", (224, 224)],
    ["convnext_tiny", (224, 224)],
    ["vgg16", (224, 224)],
    ["mobilenetv3_large_100", (224, 224)],
    ["maxvit_small_tf_224", (224, 224)]
]


# Function to get current process memory usage in GB
def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)  # Convert to GB

def load_images(image_dir, batch_size, image_size, max_images=1000):
    """
    Load images from a directory and prepare them for model input
    Returns a batch of preprocessed images
    
    Args:
        image_dir: Directory containing images
        batch_size: Size of each batch
        image_size: Tuple of (height, width) for model input
        max_images: Maximum number of images to load
    """
    # Standard preprocessing for pretrained models
    preprocess = transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.14)),  # Resize to slightly larger than needed
        transforms.CenterCrop(image_size),             # Then crop to exact size
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
    
    print(f"\nFound {len(image_paths)} images in {image_dir}")
    
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
    
    print(f"\nCreated {len(batches)} batches of size {batch_size}")
    return batches

def benchmark_model(model_info, device, image_dir, batch_sizes):
    """
    Benchmark a single model across different batch sizes using real images
    Returns a dictionary with timing and memory results
    
    Args:
        model_info: List containing [model_name, image_size]
        device: Device to run on ("cpu" or "cuda")
        image_dir: Directory containing images
        batch_sizes: List of batch sizes to test
    """
    model_name, image_size = model_info
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nTesting {model_name} with batch size {batch_size} on {device}")
        
        try:
            # Load images with the correct size for this model
            image_batches = load_images(image_dir, batch_size, image_size)
            
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
                print(f"\nUsing first batch for remaining measurements (have {num_batches}, need {MEASUREMENT_ITERATIONS})\n")
            
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
            single_image_latency = avg_latency/batch_size
            
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
                "input_size": f"{image_size[0]}x{image_size[1]}",
                "batch_size": batch_size,
                "latency_ms": avg_latency,
                "latency_std_ms": std_latency,
                "single_image_latency_ms":single_image_latency,
                "gpu_memory_base_gb": start_gpu_memory,
                "gpu_memory_used_gb": gpu_memory_used,
                "cpu_memory_base_gb": start_cpu_memory,
                "cpu_memory_used_gb": cpu_memory_used,
                "params_millions": param_count,
                "device": device
            })
            
            # Clean up
            del model, image_batches
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            print(f"\n{model_name} Test finished!\n")
            print("=="*40, "\n")
                
        except Exception as e:
            print(f"Error benchmarking {model_name} with batch size {batch_size}: {str(e)}")
            results.append({
                "model": model_name,
                "input_size": f"{image_size[0]}x{image_size[1]}",
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

def run_all_benchmarks(image_dir, device, batch_sizes):
    """
    Run benchmarks for all models on the specified device
    
    Args:
        image_dir: Directory containing images
        device: Device to run on ("cpu" or "cuda")
    """
    all_results = []
    
    for model_info in tqdm(MODELS, desc=f"Benchmarking on {device}"):
        results = benchmark_model(model_info, device, image_dir, batch_sizes)
        all_results.extend(results)
            
    return all_results


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Benchmark vision transformer models with real images")
    parser.add_argument("--image_dir", type=str, required=True, 
                        help="Directory containing images for benchmarking")
    parser.add_argument("--run_on_gpu", action="store_true", 
                        help="Run benchmarks on GPU instead of CPU (default: run on CPU)")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1],
                        help="Batch sizes to benchmark (default: 1, 4, 8, 16, 32)")
    
    return parser.parse_args()

def run_and_save_benchmarks(args):
    """
    Run all benchmarks and save results
    """
    print("=="*40)
    
    batch_sizes = args.batch_sizes
    print(f"\n Using custom batch sizes: {batch_sizes}")
    
    # Determine device to use
    if args.run_on_gpu and torch.cuda.is_available():
        device = "cuda"
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        if args.run_on_gpu and not torch.cuda.is_available():
            print("GPU requested but CUDA is not available. Running on CPU instead.")
        else:
            print("Running on CPU")
    print("=="*40)
    # Get system info
    system_info = {
        "cpu": platform.processor(),
        "ram": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "image_dir": args.image_dir,
        "device_used": device
    }
    
    print(f"\nSystem info: {system_info}")
    print("=="*40)
    
    # Run benchmarks on the selected device
    benchamrk_result = run_all_benchmarks(args.image_dir, device, batch_sizes)

    results_df = pd.DataFrame(benchamrk_result)

    print("=="*40)
    print("Saving results to local storage.")
    
    # Save raw results
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_df.to_csv(f"vit_benchmark_results_{timestamp}_{device}.csv", index=False)
    print(f"vit_benchmark_results_{timestamp}_{device}.csv")
    
    # Create output directory for JSON results
    os.makedirs("results", exist_ok=True)
    
    # Save system info
    with open(f"results/system_info_{timestamp}_{device}.json", "w") as f:
        import json
        json.dump({"system_info":system_info, "benchamrk_result":benchamrk_result}, f, indent=2)
    print(f"results/system_info_{timestamp}_{device}.json")
    print("=="*40)
    
    return results_df, system_info

if __name__ == "__main__":
    args = parse_arguments()
    results_df, system_info = run_and_save_benchmarks(args)

"""
Example command :
python3 bench.py --image_dir /workspace/dataset/images/test --run_on_gpu --batch_sizes 1 2 4
"""