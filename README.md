# Sartorius cell classification with Convolutional Neural Network (CNN)-2

## Project Overview
### Introduction
This project aims to classify images of various cellular classes provided by Sartorius using Convolutional Neural Networks (CNNs). The dataset comprises images from 9 distinct cellular classes: `A172`, `BT474`, `BV2`, `Huh7`, `MCF7`, `RatC6`, `SHSY5Y`, `SkBr3`, and `SKOV3`.
### Dataset Structure
The dataset is organized into two main directories: `livecell_train_val_images` and `livecell_test_images`. Each of these directories contains images of the aforementioned cellular classes. Specifically, within each directory, there are 9 subdirectories corresponding to the cell types mentioned earlier.
### Data Access
The data is accessible and downloadable from [here](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data).

**Reference:**
- Addison Howard, Ashley Chow, CorporateResearchSartorius, Maria Ca, Phil Culliton, Tim Jackson. [Sartorius - Cell Instance Segmentation](https://kaggle.com/competitions/sartorius-cell-instance-segmentation). Kaggle (2021).

### Tools and Libraries Used
- Python 3.6.15
- numpy 1.19.2
- opencv 3.4.2
- scikit-learn 0.24.2
- keras 2.3.1
- matplotlib 3.3.4
- seaborn 0.11.2

### An example of nine types of cells
<img src="https://github.com/mohammadhosseinparsaei/Cells-classification-Sartorius/blob/main/The%20last%20image%20of%20each%20cell.png" alt="images" width="700"/>

### Model Performance

#### Normalized Confusion Matrix
<img src="https://github.com/mohammadhosseinparsaei/Cells-classification-Sartorius/blob/main/normalized_confusion_matrix.png" alt=" Confusion_Matrix_1" width="600"/>

#### Confusion Matrix without Normalization
<img src="https://github.com/mohammadhosseinparsaei/Cells-classification-Sartorius/blob/main/confusion_matrix_without_normalization.png" alt=" Confusion_Matrix_2" width="600"/>

#### Accuracy Plot
<img src="https://github.com/mohammadhosseinparsaei/Cells-classification-Sartorius/blob/main/accuracy_plot.png" alt="accuracy" width="400"/>

#### Loss Plot
<img src="https://github.com/mohammadhosseinparsaei/Cells-classification-Sartorius/blob/main/loss_plot.png" alt="loss" width="400"/>
