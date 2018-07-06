# Production ready Data-Science with Python and Luigi

## Description

The rapidly growing size of data has broadened the role of data scientists. Instead of creating models for one-off analysis, it's becoming common to use prototypes in real-world applications. Spotify Luigi helps to enable such scenarios, without having to learn a whole bunch of new technologies. We will explore Luigi's concepts and learn how to code solutions for common problems in data pipelines. 

## Abstract
You will build a robust ML-Pipeline from scratch with Spotify Luigi
You will discover and learn how to utilize some of Luigis modules to avoid boilerplate code
You will learn how easy it is to transition prototyped models to production ready code

## Luigi Pipeline

### Training

1. Download

1. Create Generator

1. Create untrained Model

1. Train Model

1. Export to TensorFlow-Serving

### Inference

1. Check Model

1. Check Generator

1. Predict / Inference

PYTHONPATH='.' luigi --module 00_training_pipeline Download --version 1 --local-scheduler