# Trasfer learning with synthetic data
## Optimizing Deep Learning Performance with Limited Multivariate Time Series Data

## Table of Contents
1. Introduction
2. Motivation
3. Methodology
4. Datasets
5. Experimental Setup
6. Results and Analysis
7. Code Availability
8. Conclusion
9. Introduction

This repository contains the code and resources for my graduation project, which aims to optimize deep learning performance in the context of limited multivariate time series data. The project explores the challenge of scarce data in real-world applications of deep learning and investigates whether a transfer learning approach can improve the efficiency and robustness of predictive models.

## Motivation
In many real-world scenarios, limited availability of data poses significant challenges to the effectiveness of deep learning models. This project addresses this issue by employing a transfer learning technique, specifically using a model pretrained on synthetic data and fine-tuned on the original dataset. The goal is to surpass the performance of models trained solely on the original data.

## Methodology
The project utilizes 14 datasets from the UAE repository, a well-established benchmark for academic research in time series classification. Two types of deep learning architectures, namely Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN), were selected as baseline models. These models were trained using various hyperparameters.

To explore the transfer learning approach, synthetic data was generated using a Generative Adversarial Network (GAN). The GAN was trained on different portions of the original training set, producing synthetic data for the pretraining phase. Subsequently, the pretrained models were fine-tuned on the original dataset using the same hyperparameters.

Statistical tests such as the t-test and Welch test were conducted to examine the significance of mean differences between the models trained solely on original data and those pretrained with synthetic data. Normality and equal variance assumption tests were performed to validate the findings.

## Datasets
The project utilizes 14 datasets sourced from the UAE repository, which is widely recognized as a benchmark for time series classification research. These datasets cover a range of real-world scenarios and provide a comprehensive basis for evaluating the effectiveness of the proposed transfer learning approach.

## Experimental Setup
The baseline models, including CNN and RNN architectures, were trained on the original datasets using various hyperparameters. These models served as the comparison for the pretrained models.

The synthetic data generation involved training a GAN on different subsets of the original training set. This process resulted in synthetic data that captured the underlying patterns present in the original data. The pretrained models were then fine-tuned using the synthetic data, along with the original hyperparameters.

## Results and Analysis
The project's research produced a comprehensive statistical analysis, highlighting the effectiveness of the proposed transfer learning approach. Key factors such as learning rate, sequence length, number of classes, and features were identified as significant contributors to model robustness and performance.

The experiments demonstrated notable improvements achieved using the two-phase training approach, with up to a 37.5% increase in F1 score and other metrics across various data aggregations. Remarkable enhancements were consistently observed on specific datasets, suggesting the need for further exploration of data generation techniques to better capture underlying interconnections and complexities.

## Code Availability
The code developed for this project is available for future research. It includes modular and generic experimentation code that allows for the incorporation of additional models, hyperparameter search methods, and settings, given sufficient computational resources. The repository is open to other researchers, promoting collaboration and advancements in the field.

## Conclusion
This graduation project addressed the challenge of limited multivariate time series data in deep learning applications
