## MDProp: Official Implementation of ACML 2023 Paper: "Advancing Deep Metric Learning With Adversarial Robustness"

### Table of Contents
1. [Requirements](#requirements)
2. [Usage](#usage)
3. [Configuration](#configuration)
4. [Supplementary Material](#supplementary-material)

---

### Requirements

The following software packages are required to use this repository:

- Python 3.6 or higher
- PyTorch 1.9 or higher
- torchvision
- numpy
- scikit-learn

---

### Usage

#### Training and Evaluation Scripts

Inside the parent directory, you'll find scripts for training and evaluating Deep Metric Learning models using MDProp and the multisimilarity loss. MDProp is implemented with three separate batch normalization layers, optimized for clean data, single-targeted adversarial data, and multi-targeted adversarial data. Optimal results are achieved with 5 attack targets in multi-targeted attacks.

**Steps for Usage:**

1. Download the CUB200, CARS196, and SOP datasets and extract them to a suitable directory.
2. Download the official PyTorch ResNet50 parameters from [here](https://download.pytorch.org/models/resnet50-19c8e357.pth) and place them in the `params` folder.
3. Update the data paths (`--source_path`) in `train.sh`.
4. Run the `train.sh` script. Results will be saved in the `Training_Results` folder.

**Note**: There may be minor variations in results due to GPU architecture, the number of GPUs used, and dataloader workers.

For parameter descriptions, consult the `parameters.py` file.

---

### Configuration
The repository and the results are configured for an embedding size of 128.

---

### Supplementary Material
The `MDProp_Supplmentary_Material.pdf` file provides supplementary explanations cited in the paper.

---

Feel free to raise issues or submit pull requests for any questions or improvements.

