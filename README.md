# Stable-Diffusion-Core

`Stable-Diffusion-Core` is a comprehensive implementation of the Stable Diffusion model from scratch. This repository includes the core components necessary for training and deploying diffusion models, including encoders, decoders, CLIP encoding, and U-Net architectures.


### Installation and Setup

1. Clone the git repo :
   ```bash
   git clone https://github.com/AniruthSuresh/Stable-Diffusion-Core.git
   cd Stable-Diffusion-Core

2. Create the conda environment (tennis in this case) using the provided `env.yml` file and activate it:
   
   ```bash
   conda env create -f env.yml
   conda activate diffusion

3. To get the final output :
   ```bash
   cd src
   python3 check.py

## Download Weights and Tokenizer Files

To set up the Stable Diffusion model, you need to download and save the necessary files in the `data` folder:

1. **Download Tokenizer Files**:
    - **`vocab.json`** and **`merges.txt`** can be downloaded from [Hugging Face Tokenizer Files](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main/tokenizer).
    - Save these files into the `data` directory of your project.

2. **Download Model Weights**:
    - **`v1-5-pruned-emaonly.ckpt`** can be downloaded from [Hugging Face Model Weights](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main).
    - Save this file into the `data` directory of your project.
  
## Complete Architecture
![image](https://github.com/AniruthSuresh/Stable-Diffusion-Core/blob/main/data/architecture.png)
