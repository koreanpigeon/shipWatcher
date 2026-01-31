## The shipWatcher
My first Machine Learning(Computer Vision) project using **PyTorch** to detect unidentified personnel / vehicle approaching a Navy vessel at night.
This project aims to solve a problem that I, and many others in the Navy today still suffer from -- Gangway Watch.

## Description of project
I served in the Republic of Korea Navy onboard a Guided-missile Frigate for 20 months as a conscript.
One of the most tiring tasks was Gangway Watch, a naval tradition where the gangway was to be guarded 24/7 by 2 crew with rotating shifts. 
Watching my surroundings intensely at 3am to not get caught on surprise by the fleet's officer on-duty on his impromptu patrols took a huge toll on my body.
I first started learning ML in the Navy, and I thought it would be meaningful to end my service with a project that captured my personal experiences.

## Current Status: Training & Data Collection**
This project is currently in the **active development phase**. 
I am focusing on improving model accuracy by 
* Addressing domain gaps between web-scraped images and real-world video frames
* Training models with images of various weather conditions that a Naval Base is susceptible to due to its proximity to the sea.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/kimjiminnus/shipWatcher.git
cd shipWatcher
pip install -r requirements.txt
```

### 2. Local file organisation structure
```bash
shipWatcher/
├── data/
│   ├── train/          # Training Images
│   │   ├── Empty/
│   │   ├── Person/
│   │   └── Vehicle/
│   └── val/            # Validation Images
│       ├── Empty/
│       ├── Person/
│       └── Vehicle/
├── models/             # Where 'shipWatcher.pth' will be saved
├── src/                # Source code (train.py, inference.py, data_preprocessing.py, tune_hyperparams.py, video.processing.py)
└── requirements.txt
```

### 3. Training shipwatcher from scratch & saving state_dict
```bash
python src/train.py
```

### 4. Testing your own image files on the shipWatcher 
```bash
python src/inference.py
```


## The Tech Stack
* **Language:** Python 3.x
* **Framework:** PyTorch
* **Computer Vision:** Torchvision, OpenCV
* **Model:** ResNet-18 with a customised 3-neuron output layer (Transfer Learning)

## Challenges I'm Solving
* **Data Scarcity:** Finding suitable datasets of images in a Naval Base proves to be challenging due to Operational Security(OPSEC)
* **Class Imbalance:** My "Empty" class is smaller than "Vehicle" or "Person." I am implementing **Weighted Random Sampling** to prevent model bias.
* **Accuracy:** Current accuracy is limited by background noise. I am working on **Hard Negative Mining** to reduce false positives on poles and shadows.

## 🚀 Future Roadmap
1. Once the model is trained with appropriate data and validated & tested on saved videos, adjust cv2.VideoCapture parameter to process live video feed.
2. Implement a Confusion Matrix for better error analysis.
3. Export to CoreML and Streamlit to share & deploy model.
