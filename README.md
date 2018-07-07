# Production ready Data-Science with Python and Luigi

This repo contains the slides, notebooks and example code for the PyData Berlin 2018 tutorial.

## Code

The example luigi pipeline is under "pipelines". It can be started by running

````bash
PYTHONPATH='.' luigi --module 00_training_pipeline Export --dataset-version 1 --model-version 1
```

The notebooks contain the tasks and a few examples for presentation purposes.

## Slides

The slides are accessible via the Jupyter Notebook "Production-Ready-Datascience.ipynb" as well as
the PDF "presentation.pdf".