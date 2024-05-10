# Sartorius cell classification with Convolutional Neural Network (CNN)-2

## Project Overview
### Introduction
This project aims to classify images of various cellular classes provided by Sartorius using Convolutional Neural Networks (CNNs). The dataset comprises images from 9 distinct cellular classes: `A172`, `BT474`, `BV2`, `Huh7`, `MCF7`, `RatC6`, `SHSY5Y`, `SkBr3`, and `SKOV3`.
### Dataset Structure
The dataset is organized into two main directories: `livecell_train_val_images` and `livecell_test_images`. Each of these directories contains images of the aforementioned cellular classes. Specifically, within each directory, there are 9 subdirectories corresponding to the cell types mentioned earlier.
### Data Preprocessing
To create a comprehensive dataset for training, images from the 9 cellular class directories within the `livecell_train_val_images` directory were combined with images from the corresponding directories within the `livecell_test_images` directory. This consolidation resulted in a larger dataset suitable for model training. Subsequently, **10%** of the combined data was randomly sampled and set aside for **testing purposes**.
### Data Access
The data is accessible and downloadable from [here](https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data).
**Reference:**
- Addison Howard, Ashley Chow, CorporateResearchSartorius, Maria Ca, Phil Culliton, Tim Jackson. [Sartorius - Cell Instance Segmentation](https://kaggle.com/competitions/sartorius-cell-instance-segmentation). Kaggle (2021).

### Tools and Libraries Used
- Jupyter
- numpy 1.19.2
- opencv 3.4.2
- scikit-learn 0.24.2
- keras 2.3.1
- matplotlib 3.3.4
- seaborn 0.11.2
### An example of nine types of cells
![cells](https://github.com/mohammadhosseinparsaei/Cells-classification-Sartorius/blob/main/The%20last%20image%20of%20each%20cell.png)
### Model Performance
#### Loss Plot
![Loss plot](https://github.com/mohammadhosseinparsaei/Cells-classification-Sartorius/blob/main/loss_plot.png)
#### Accuracy Plot
![Accuracy Plot](https://github.com/mohammadhosseinparsaei/Cells-classification-Sartorius/blob/main/accuracy_plot.png)
#### Normalize Confusion Matrix
![Confusion Matrix](https://github.com/mohammadhosseinparsaei/Cells-classification-Sartorius/blob/main/confusion_matrix.png)
