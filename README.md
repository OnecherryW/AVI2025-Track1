# 🏆 AVI Challenge 2025 - Track2: Personality Assessment

## 🎯 Interview Performance Assessment
This project implements a deep learning model based on multimodal feature fusion that effectively integrates audio, video, and text information to predict the following personality traits:
- **Integrity** 
- **Collegiality**
- **Social versatility**
- **Development orientation** 
- **Overall hireability** 

**We achieved 1st place in AVI Challenge 2025 Track1: Personality Assessment!** This repository contains our code structure.
| **Team Name**       | **MSE (↓)** |
|---------------------|--------------------|
| **HFUT-VisionXL**    | **0.18240 (1)**    |
| CAS-MAIS             | 0.18510 (2)        |
| ABC-Lab              | 0.19394 (3)        |
| The innovators       | 0.20749 (4)        |
| HSEmotion            | 0.22150 (5)        |
| USTC-IAT-United      | 0.24828 (6)        |
| DERS                 | 0.25540 (8)        |
| DERS                 | 0.25586 (9)        |



## 🏗️ Project Structure

```
AVI-track2/
├── main.py                 # main script
├── main.sh                 # main script
├── readme                  # Project documentation
├── model/                  # Model implementation
│   ├── vote_model
|   |   └── M_model.py  
│   └── baseline_model.sh      
├── dataset/                # Dataset processing
│   └── baseline_dataset2_vote.py 
├── Features/               # downloaded features
│   ├── audio/               # Audio features
│   ├── video/               # Video features
│   └── text/                # Text features
├── output/
├── data/
└── requirements.txt        # Project dependencies
```

## ⚙️ Environment Requirements
```bash
git clone url_to_your_repository
cd AVI-track2
```

### Step 1: Create a new conda environment with Python 3.10 (or your preferred version)
conda create -n avi2025 python=3.10 -y

### Step 2: Activate the environment
conda activate avi2025

### Step 3: Install pip (if not already installed)
conda install pip

### Step 4: Install dependencies from requirements.txt
python -m pip install -r requirements.txt

## 🚀 Quick Start


### 1. 📋 Data Preparation

Downloaded features are required for training and testing. The dataset includes audio, video, and text features extracted from the AVI Challenge 2025 dataset.
Baidu Cloud link for downloading the features: [Quark](https://pan.quark.cn/s/5d29aa346d01) (Password: `QiuD`)

Ensure data file paths are correctly configured:
```bash
# Training data
TRAIN_CSV="path/to/train_data.csv"
VAL_CSV="path/to/val_data.csv"
TEST_CSV="path/to/test_data.csv"

# Feature directories
AUDIO_DIR="path/to/audio/features"
VIDEO_DIR="path/to/video/features"
TEXT_DIR="path/to/text/features"
```

### 2. 🏋️‍♂️ Model Training

Use the provided training script:
```bash
cd AVI-track2
bash main.sh
```

### 3. 🧪 Model Testing
Note: To achieve better generalization performance, we recommend that only text features be used during testing.
```bash
bash ./scripts/test.sh
```
Ensure data file paths are correctly configured:
```bash
# Training data
ARGS_JSON="path/to/args.json"
SUBMISSION_CSV="path/to/submission.csv"
TEST_CSV="path/to/test_data.csv"

# trait to predict 
NOTE: trait should be same as label_col in args.json
TRAIT="Honesty-Humility"  # Honesty-Humility, Extraversion, Agreeableness, Conscientiousness
```

### 📏 Loss Function
-  Mean Squared Error (MSE) loss


## 📋 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

If you have any questions or suggestions, please contact the project maintainers (HFUT-VisionXL).

---

⚠️ **Note**: This project is for academic research purposes only. Please comply with relevant data usage agreements and competition rules.

## 🙏 Acknowledgments

- 🏆 Thanks to the AVI Challenge 2025 organizers
- 🤗 Thanks to the developers of [MERtools](https://github.com/zeroQiaoba/MERTools) for their excellent open-source tools that supported our data preprocessing.
