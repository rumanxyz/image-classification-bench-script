## Required Packages:
- PyTorch and torchvision
- timm
- Pillow

## Model Compatibility
This script is designed to work with models available in the timm library. If a model not present in timm is provided, it will result in an error.

## Adding a New Model
You can benchmark any model, even those not listed in the `MODELS` list. To add a model, simply include its name and the corresponding image size in the `MODELS` list. For example:
- `["maxvit_small_tf_224", (224, 224)]` to add the `maxvit_small_tf_224` model.
- `["convnext_tiny", (224, 224)]` to add the `ConvNext` model.

## Running the Script

To run the script, use the following command:
```bash
python3 bench.py --image_dir /path/to/images --run_on_gpu --batch_sizes 1 2
```
Where:
- `--image_dir` specifies the path to the directory containing the images for testing.
- `--run_on_gpu` is an optional flag (default: false). When provided, the benchmark will run on the GPU if available.
- `--batch_sizes` is an optional argument (default: 1). You can specify different batch sizes to test the model with.

### Example: Benchmarking on GPU
To run the benchmark on a GPU, use:
```bash
python3 bench.py --image_dir /path/to/images --run_on_gpu --batch_sizes 1 2 4 6 8 16
```

### Example: Benchmarking on CPU
To run the benchmark on the CPU, use:
```bash
python3 bench.py --image_dir /path/to/images --batch_sizes 1
```

## Example in Google Colab:
For an interactive notebook, check out the following link:
[Colab Notebook Example](https://colab.research.google.com/drive/13ugsERVfnJbWbPlg7DGlVZtSe9xOIQm7?usp=sharing)
