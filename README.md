This repository contains a custom-built Deep Learning pipeline designed to analyze and predict sequential stock market trends. Moving beyond standard LSTM architectures, this project leverages a Transformer Neural Network with Multi-Head Attention to dynamically weigh the importance of historical price fluctuations within a 90-day lookback window.
Built entirely in Python using TensorFlow/Keras, the architecture tackles several common bottlenecks in quantitative financial modeling:
High-Speed Data Ingestion: Implements a custom Numpy caching system to reduce sequential CSV loading times from minutes to milliseconds.
Dynamic Normalization: Uses Min-Max Window Scaling to preserve relative momentum shapes rather than anchoring to fixed price endpoints.
Algorithmic Stability: Re-architected dense layers utilizing LeakyReLU, BatchNormalization, and strict Dropout mechanisms to prevent the "Dying ReLU" model collapse common in fractional percentage predictions.
Class Balancing: Applies mathematical class weighting to handle market bias and force the model to evaluate downward trends accurately.
Strict Evaluation: Pipeline includes a sealed Train/Validation/Test split ensuring zero data leakage, outputting precision, recall, and F1-score matrices.
