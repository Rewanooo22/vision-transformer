# Image Generation with Vision Transformers

A Google Colab assignment that builds a lightweight **ViT-VAE**: a variational autoencoder with a Vision Transformer encoder. It's trained on the [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) dataset to reconstruct and generate 64×64 flower images.

## What the notebook covers

1. Kaggle API setup and dataset download
2. What Vision Transformers are and how they can generate images
3. CNN-based vs transformer-based image generation
4. Loading, preprocessing and visualising the images (64×64)
5. **Model:** ViT encoder → 256-dimensional latent space → decoder
6. VAE loss (reconstruction + KL divergence) and a training loop (30 epochs, batch size 64)
7. Loss curves, reconstructions, and new images sampled from the latent space
8. Why the results aren't perfect, plus the advantages and drawbacks of ViTs for generation

## Files

| File | Description |
|---|---|
| `improved_vit_image_generation_assignment_colab_v2.ipynb` | **Main notebook**: the full, improved version |
| `rewan_vit_image_generation_assignment_colab.ipynb` | Earlier, shorter version |
| `ViT_Image_Generation_Report.docx` | Written report |

## Run it

1. Open `improved_vit_image_generation_assignment_colab_v2.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set the runtime to GPU (Runtime → Change runtime type).
3. Have your Kaggle API token ready (`kaggle.json`, from Kaggle → Account → Create API Token). The notebook asks you to upload it.
4. Run all cells.

## Tech

PyTorch · torchvision · NumPy · Matplotlib · Pillow · Kaggle API
