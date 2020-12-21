# breast_histopathology 
Invasive Ductal Carcinoma is one of the most frequent types of breast cancer.   
The Invasive Ductal Carcinoma develops initially in milk duct's. These paths are responsible for transport the milk, which is produced in the lobes, to the nipple. As the cancer grows it tends to invade other areas of the breast, usually its tissues and lobes.  
The cancer cells are gradually surrounding the entire duct area and in last phase, the cells end to invade the breast tissues.

## Data
The Breast Histopathology presents a binary classification problem composed by 277524 samples.  
The samples correspond to scanned images of breast tissues, regarding several patients. The objective of the problem includes the correct classification of tissues with IDC and without IDC.  
The samples are available in RGB format and have dimensions of 50*50 pixels, respectively width and length.

## Limitations
The main limitations of this benchmark are:
* Moderate unbalanced classes (The distribution of the samples of the two classes shows a disproportion of 2.5-1 samples, respectively Without IDC and With IDC classes);
* Problem with high complexity. In spite of the existence of large number of samples, the distinction between the two classes is not trivial;
* The models tend to be tedious, that is, they are much more likely to correctly classify samples from the most balanced class, thus resulting in very different performances between the two classes.

## What this project offers
* Disponibilization of a Jupyter notebook with problem pre-analysis;
* Several techniques are applied to reduce the main limitations of the problem, such as: Random Undersampling (since there is a high amount of information available, the disproportion of the classes can be mitigated by randomly eliminating samples from the most balanced class), Cost-Sensitive-Learning (More severe penalty for less balanced class errors) and Data Augmentation (reduce overfitting);
* It implements and uses four convolutional architectures for the consequent resolution of the problem: AlexNet, VGGNet, ResNet and DenseNet;
* Use of PSO algorithm to optimize the structure and other hyperparameters of different convolutional architectures;
* Application of the ensemble technique to improve the performance obtained, individually, by the architectures (combining the probabilistic distributions of the different architectures - average);

## How can I use it
1. Clone Project: git clone 
2. Install requirements: pip install -r requirements.txt
3. Check config.py file, and redraw the configuration variables used to read, obtain and divide the data of the problem, and variables that are used for construction, training and optimization of the architectures.
   * Samples of problem are readed from "../breast_histopathology/input/breast-histopathology-images/IDC_regular_ps50_idx5/" folder, example path of one sample: "../breast_histopathology/input/breast-histopathology-images/IDC_regular_ps50_idx5/8863/0/8863_idx5_x51_y1251_class0.png" (8863 is the patient_id and 0 folder contains all without IDC samples of this patient) --> this is an example that you need to pay attention and redraw before use project;

### Results - Breast Histopathology:
| Model | Memory | Macro Average F1Score | Macro Average Recall | Accuracy | File | 
|---|---|---|---|---|---|
| AlexNet | 14,6 MB | 90.2% | 89.7% | 92.1% | [AlexNet h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Breast_Histopathology/alex_net_oficial.h5?raw=true) |
| VGGNet | 13,1 MB | 90.5% | 89.6% | 92.4% | [VGGNet h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Breast_Histopathology/vggnet_oficial.h5?raw=true) |
| ResNet | 29,9 MB |  90.4% |  90.2% | 92.2% | [ResNet h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Breast_Histopathology/resnet_oficial.h5?raw=true) |
| DenseNet | 10,3 MB | 89.8% |  89.4% | 91.7% | [DenseNet h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Breast_Histopathology/densenet_oficial.h5?raw=true) |
| Ensemble Average All Models | 22,8 MB | 91.1% | 90.7%  | 92.9% | [Ensemble h5 File](https://github.com/bundasmanu/ProjetoMestrado/blob/master/arquiteturas_otimizadas/Breast_Histopathology/ensemble_all.h5?raw=true) |

## Data Access
https://www.kaggle.com/paultimothymooney/breast-histopathology-images

## Licence
GPL-3.0 License  
I am open to new ideas and improvements to the current repository. However, until the defense of my master thesis, I will not accept pull request's.
