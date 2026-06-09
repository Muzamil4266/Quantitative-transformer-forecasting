🚀 Quantitative Time-Series Trend Forecasting Using a Custom Multi-Head Attention Transformer Network

A state-of-the-art, end-to-end deep learning pipeline built from the ground up in Python utilizing TensorFlow and Keras to extract, normalize, and predict sequential financial asset trends.

Moving away from standard recurrent neural network architectures (LSTMs/GRUs), this project implements a customized Transformer Layer with Multi-Head Attention paired with specialized data engineering solutions to handle the low signal-to-noise ratio (SNR) characteristic of historical financial markets.

---

📊 Project Architecture Overview

Predicting market trends is notoriously difficult due to non-stationarity, market noise, and changing macroeconomic regimes. Traditional networks like LSTMs often suffer from vanishing gradients or severe recency bias over extended context windows.

This repository introduces a standalone, optimized deep learning network structured into two key conceptual compartments:

🕵️ The Detective (Sequence Feature Encoding via Multi-Head Attention)

Processes a rolling 90-day historical window. By mapping query, key, and value vectors across multiple attention heads, the model learns to identify complex mathematical shapes (such as support floors, resistance ceilings, or high-volatility spikes) regardless of where they occur in the timeline.

📈 The Accountant (Stabilized Multi-Layer Perceptron Block)

Flattens the contextual embeddings from the attention block and processes them through an aggressively regularized feedforward network utilizing batch normalization, dropout, and leaky activation functions to determine the mathematical probability of a 30-day forward-looking directional move.

---

🛠️ Key Engineering Challenges & Solutions

⚡ 1. High-Speed Local Data Caching Pipeline

Iterating over decades of raw, multi-file daily market data in CSV formats is a significant computational bottleneck. Generating historical "flashcards" involves slicing temporal windows across dozens of files, resulting in redundant disk I/O operations.

Solution:

Engineered a robust, automated NumPy caching protocol ("nifty_cache_X.npy" and "nifty_cache_y.npy").

On the first execution, raw CSV assets are parsed, processed, and written to compressed binary files. Subsequent script runs instantly bypass the I/O bottleneck by loading data directly into system memory, decreasing dataset compilation times from minutes to milliseconds.

---

📉 2. Solving the Endpoint Normalization Trap

A common flaw in financial data engineering is anchoring lookback windows to a single fixed point (such as dividing a 90-day sequence by the price on Day 90). Mathematically, this forces the final value of every input feature vector to equal exactly "1.0".

This endpoint pinning blinds the network to localized momentum or sharp rallies/crashes occurring directly on the prediction boundary.

Solution:

Implemented localized Min-Max Window Scaling.

For every individual 90-day sliding window, the historical prices are mapped dynamically between "0.0" and "1.0" based on the strict absolute minimum and maximum values within that specific 90-day segment.

This preserves the relative structural geometry, rate of change, and momentum vectors of recent market price actions.

# Min-Max Scaling Formula Implemented per Window Sequence

min_price = np.min(past_90_days)
max_price = np.max(past_90_days)

normalized_X = (past_90_days - min_price) / (max_price - min_price)

---

🧠 3. Mitigating Structural Model Collapse (The Dying ReLU Problem)

Financial price changes are incredibly fractional. Passing small, normalized variances through a standard Rectified Linear Unit (ReLU) often leads to dead nodes.

If neurons drop below zero during backpropagation, their gradients become permanently zeroed out, forcing the model to stop updating and default to guessing the statistical mean.

Solution:

Re-architected the feedforward block using LeakyReLU activations with an explicit alpha threshold of "0.1".

This allows a tiny, non-zero gradient to pass through even when a node is inactive, ensuring continuous gradient flow.

This was paired with interleaved Batch Normalization to smooth the optimization landscape and Dropout layers to prevent the model from over-indexing on historical noise.

---

⚖️ 4. Handling Real-World Market Biases via Algorithmic Class Weighting

Historical market indexes inherently display long-term structural upward trends.

Left uncorrected, a neural network will learn that it can achieve basic accuracy by simply predicting "UP" on every sample, completely ignoring downward movements.

Solution:

Integrated an automated class balancing utility using Scikit-Learn to calculate dynamic inverse weights for target labels based on data distribution.

These metrics are fed directly into the binary cross-entropy loss calculation during the forward pass, artificially amplifying the loss penalty when the network incorrectly categorizes rarer market downward movements.

---

📐 Mathematical Pipeline Architecture

[Input Vector: Raw CSV Prices]
          │
          ▼
(Lookback: 90 Days | Target Horizon: 30 Days)

[Rolling Window Extraction]
          │
          ▼

[Min-Max Local Window Scaling]
          │
          ▼

[Feature Matrix Shape: (Samples, 90, 1)]
          │
          ▼

[Multi-Head Attention Layer]
(4 Attention Heads | Key Dimension: 64)
          │
          ▼

[Flattened Embedding Context Vector]
          │
          ▼

[Dense Block 1]
          │
          ▼

[Batch Normalization]
          │
          ▼

[LeakyReLU (α = 0.1)]
          │
          ▼

[Dropout (0.3)]
          │
          ▼

[Dense Block 2]
          │
          ▼

[Batch Normalization]
          │
          ▼

[LeakyReLU (α = 0.1)]
          │
          ▼

[Dropout (0.2)]
          │
          ▼

[Dense Block 3]
          │
          ▼

[Batch Normalization]
          │
          ▼

[LeakyReLU (α = 0.1)]
          │
          ▼

[Output Projection Layer]
          │
          ▼

[Sigmoid Activation]
          │
          ▼

[Binary Probabilistic Trend Target]

---

🚀 Installation & System Configuration

📦 1. Prerequisites & Environment Setup

Ensure your local development environment has Python 3.8+ installed along with necessary CUDA dependencies if executing on a dedicated GPU framework.

Clone the repository locally:

git clone https://github.com/YOUR_USERNAME/quantitative-transformer-forecasting.git

cd quantitative-transformer-forecasting

---

🔧 2. Install Dependencies

Install all necessary data science, matrix manipulation, and deep learning dependencies.

---

📁 3. Setting Up Data Paths

Open "train_transformer.py" and configure the absolute filepath string pointing to the local directory where your raw market CSV resource files are stored:

FOLDER_PATH = r"C:\Your\Local\Path\To\Nifty-Training-dataset"

---

🏋️ Training Protocol

The training architecture enforces an isolated 3-Way Data Split to safeguard the network against any form of lookahead bias or data leakage.

The entire timeline array is divided into three distinct sets:

📚 70% Pure Training Set

Used exclusively to adjust the network weights via backpropagation.

📝 10% Validation Practice Quiz

Utilized at the end of every training epoch to monitor out-of-sample loss variance and trigger optimization callbacks.

🎓 20% Sealed Final Exam Pile

Completely isolated from the data preprocessing pipeline and only unsealed after the training cycle is terminated to verify true generalized performance.

To initialize the compilation, processing, and training sequence, execute the master file:

transformer_forecasting.py

---

🛡️ Automated Training Safeguards

💾 Model Checkpointing

Monitors validation loss continuously.

The script automatically saves the model weights to:

best_nifty_transformer.keras

only if the validation loss achieves a new historical low, ensuring that post-epoch training degradation or overfitting is skipped.

⏹️ Early Stopping

Programmed with a patience threshold of 8 epochs.

If validation loss plateaus or begins climbing due to noise memorization, training halts automatically and reinstates the historical optimal weight matrices.

---

📈 Empirical Results & Final Evaluation

Once training successfully terminates, the pipeline unseals the completely unseen 20% Final Exam Pile to compute robust out-of-sample classification analytics.

📊 Training Progress Metrics

The optimization curve tracks model performance and learning stability across training milestones.

---

🎯 Unsealed Performance Matrix

The model outputs a comprehensive suite of evaluation metrics specifically focused on directional categorization accuracy rather than standard regression variances.

📌 Metric Interpretations

Directional Accuracy

Demonstrates true out-of-sample generalization.

Given the immense noise of index funds, achieving an accuracy floor above the 50% baseline indicates the successful capture of persistent mathematical momentum signals.

Precision & Recall Balance

Highlights the effectiveness of the class weighting algorithm, proving the model does not biasedly over-predict upward market trends.

Confusion Matrix Breakdown

Validates the absolute distribution of:

- True Positives (TP)
- False Positives (FP)
- True Negatives (TN)
- False Negatives (FN)

demonstrating empirical resilience against model collapse.

---

🏛️ Core Takeaways for Quantitative Modeling

Developing this pipeline highlights a profound reality of quantitative finance and complex machine learning architectures:

«The quality of prediction is capped by data engineering, not model scale.»

Big neural network models do not extract predictive alpha if the data is poorly structured or implicitly masked by flawed normalization choices.

Fixing the window scaling mechanics was what allowed the Multi-Head Attention layer to start identifying meaningful pattern shapes.

In highly efficient financial markets, achieving a consistent, out-of-sample directional edge between 53% and 55% represents an exceptional foundation for risk-mitigated algorithmic exploitation.
