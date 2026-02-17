# CanisCare — Dog Skin Disease Classifier

Brief MATLAB project for classifying dog skin lesions using transfer learning on DenseNet-201, including training scripts, an interactive GUI, EDA utilities, and owner-facing advice text.

**Status:** Working prototype — model saved as `DogSkinDenseNet201.mat` (trained network, input size, classes).

**Table of Contents**
- **Overview**: What this project does
- **Requirements**: MATLAB toolboxes and environment
- **Quick Start**: How to run training, GUI, and prediction
- **Files**: Short description of key files
- **Notes & Tips**: Important configuration and troubleshooting
- **License & Contact**

**Overview**
- **Purpose**: Provide a screening tool to classify common dog skin lesions from images and give preliminary owner advice. Not a replacement for veterinary diagnosis.
- **Approach**: Transfer learning using DenseNet-201, with data augmentation and class-weighted loss to handle imbalance.

**Requirements**
- **MATLAB** (R2019b or newer recommended)
- **Toolboxes**: Deep Learning Toolbox, Image Processing Toolbox
- **Optional**: Deep Learning Toolbox Model for DenseNet-201 (provides `densenet201` function)
- Sufficient GPU is recommended for training (Parallel Computing Toolbox + GPU support), but CPU training works for small experiments.

**Quick Start**
1. Place your dataset in a folder where subfolders are disease class names, e.g.:

```
C:\Users\<you>\Desktop\DOG SKIN DISEASES\
    ├── Actinic Keratosis\
    ├── Melanoma\
    ├── Seborrheic Keratosis\
    └── Vascular Lesion\
```

2. Train the model (edit dataset path inside `TrainDogSkinModel.m` first):

MATLAB interactive:
```
open TrainDogSkinModel.m
Edit `folderPath` to your dataset folder and run the script.
```

Or run from your system shell (PowerShell) if `matlab` is in PATH:
```
matlab -batch "TrainDogSkinModel"
```

Training outputs a file `DogSkinDenseNet201.mat` which contains: `trainedNet`, `inputSize`, and `classes`.

3. Run the GUI for single-image prediction:

Open MATLAB and run:
```
DogSkinApp
```

In the GUI, click `Load Image from Disk` to choose an input image. The app will display a prediction, confidence, owner advice (from `getOwnerAdvice.m`), and draw a lesion bounding box.

4. Use the lightweight prediction helper (if you want to integrate into another script):

- `loadAndPredict.m` contains a prediction flow (note: it expects GUI variables like `ax`, `lblResult` if used as-is). Prefer using the GUI or adapt the code for command-line use.

**Files and Purpose**
- `TrainDogSkinModel.m`: Full training pipeline — loads dataset, computes class weights, sets up augmentation, modifies DenseNet-201 for transfer learning, trains, evaluates, and saves the final model to `DogSkinDenseNet201.mat`.
- `DogSkinDenseNet201.mat`: Trained model artifact (contains `trainedNet`, `inputSize`, `classes`).
- `DogSkinApp.m`: MATLAB UIFigure GUI for loading an image, running classification, showing confidence, owner advice, and drawing a lesion bounding box. Uses `getOwnerAdvice.m` and `DogSkinDenseNet201.mat`.
- `loadAndPredict.m`: Helper function used by the GUI to load an image, preprocess, classify and display feature maps. It references GUI axes and labels when used from `DogSkinApp`.
- `DogSkin_EDA.m`: Exploratory Data Analysis script — class distribution, sample images, image-size statistics, RGB histograms, and simple feature correlation heatmaps.
- `getOwnerAdvice.m`: Returns user-friendly advice text (cell array) for predicted disease labels. The advice strings are shown in the GUI.
- `findLayersToReplace.m`: Utility that helps locate the learnable layer and classification layer in a `LayerGraph` for transfer learning and layer replacement.

**Configuration & Tips**
- Update `folderPath` in `TrainDogSkinModel.m` and `DogSkin_EDA.m` to point to your dataset folder before running.
- If you do not have the DenseNet model installed, install the compatible support package (Deep Learning Toolbox Model for DenseNet-201) or switch to another pretrained network (e.g., `resnet50`) and adjust layer names in the training script.
- Training hyperparameters (epochs, mini-batch size, learning rate) are set in `TrainDogSkinModel.m` under `trainingOptions` — tune them for your dataset size and GPU availability.
- The GUI and `loadAndPredict.m` expect `DogSkinDenseNet201.mat` to be in the same folder as the scripts; move the `.mat` file or change load paths if needed.

**Limitations & Warnings**
- This project is a screening aid only. The advice and classification should never be used as a definitive veterinary diagnosis.
- Model performance depends heavily on dataset quality and class balance. Use proper validation and consult domain experts for clinical use.

**Next Steps (suggestions)**
- Add a command-line prediction wrapper that accepts image paths and returns JSON outputs for easier integration.
- Add unit tests or a small script to validate model loading and a sample prediction.
- Package as a MATLAB App for easier distribution.

**License & Contact**
- This repository does not include an explicit license file. If you want to open-source it, add a `LICENSE` (e.g., MIT) at the project root.
- For questions or collaboration, contact the repository owner.

---
Generated README for the CanisCare Dog Skin Disease Classifier.
