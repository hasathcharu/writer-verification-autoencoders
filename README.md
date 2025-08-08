# Explainable Writer Verification with Autoencoders and Feature Fusion

Biometrics are a popular method for verifying the identity of individuals based on their unique physical and behavioral traits. This repository presents a novel approach to explainable writer verification using autoencoders as well as a new feature fusion architecture that combines line level and character level features.

This is my main contribution to the final year research project titled Exam Candidate Verification Through Handwritten Artifacts, available [here](https://github.com/hasathcharu/exam-candidate-verification). I was responsible for designing and developing Module 3 – Personalized Writer Verification with Manual Feature Extraction.

## Key Features

- Novel feature fusion architecture that combines line level and character level features.
- Autoencoder-based approach for writer verification.
- SHAP-based explanations to understand the model's predictions.
- LLM-assisted SHAP value interpretations for enhanced understandability of model decisions.

## Feature Extraction Pipeline

![Feature Extraction Pipeline](assets/feature_extraction.svg)

The feature extraction pipeline is designed to extract both line level and character level features from handwritten samples. In the preprocessing stage, the images are resized and cropped. Then a rule-removal algorithm which takes advantage of the fact that the ink is blue as well as the rules are less saturated than the ink is applied to remove the rules from the images, with minimal damage to the handwritten content. Then, gray scaling and binarization are applied. Line segmentation is done using the horizontal projection method in combination of a line cleaning algorithm that removes overlapping bits of text from adjacent lines. The character 'e' instances are detected using a YOLOv8 Small model trained to detect the character 'e' in handwritten samples. The shown line level and character level features are then extracted using a combination of image processing techniques.

## Feature Fusion and Autoencoder Architecture

![Feature Fusion and Autoencoder Architecture](assets/training_pipeline.svg)

The feature fusion architecture combines the line level and character level features extracted from the handwritten samples. The fused features are used to train a lightweight autoencoder model that learns to reconstruct the writer's handwriting style. The threshold for the reconstruction error is determined using a validation set, using Equal Error Rate (EER) as the metric. The trained autoencoder is then used to verify the identity of the writer by comparing the reconstruction error of the test sample with the threshold. These model predictions are then explained using SHAP values, which provide insights into the most important features contributing to the model's decisions. These SHAP values are then interpreted using an LLM to provide a more understandable explanation of the model's predictions.

A more detailed explanation of this work will be made available soon.