# Weather Pixels: Neural Networks from Perceptron to Backpropagation

An introductory, from-scratch NumPy project that teaches Units I–III through a tiny weather-map classification problem. The model looks at a sequence of synthetic 4×4 weather maps and predicts whether the next map is clear, rainy, or stormy.

The selected reference project studies short-range forecasts of real gridded climate variables with ConvLSTMs and external weather datasets. This project keeps the weather-map motivation while using small synthetic examples and basic fully connected networks, so the core ideas in the syllabus stay visible.

## What you will learn

| Syllabus topic | Where it appears |
|---|---|
| Brain-inspired neurons, inputs, outputs | `src/weather_nn.py`: `sigmoid`, `Perceptron`, and `MLP` |
| Binary classification and perceptron training | `Perceptron.fit` learns rain vs. no rain |
| Multiclass classification | The MLP predicts clear, rain, or storm |
| Pixel counts and weight-matrix dimensions | `make_dataset` flattens three 4×4 maps into 48 input values; layer shapes are printed by the demo |
| Feedforward and vectorized information flow | `MLP.forward` uses matrix multiplication for a whole batch |
| Loss, weights, biases, and hyperparameters | `MLP.fit` reports cross-entropy loss and updates parameters with gradient descent |
| Sigmoid backpropagation and batches | `MLP.fit` computes gradients for mini-batches using the chain rule |

## Run it

Requires Python 3.10+ and NumPy.

```bash
python -m pip install -r requirements.txt
python -m src.weather_nn
```

No network access or real weather data is needed. The generated data is for learning only and is not a real forecast.

## Project layout

```text
weather-neural-network/
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    └── weather_nn.py
```

## Model in one picture

```text
3 past maps × 4 rows × 4 columns = 48 pixel inputs
                         │
                         ▼
                48 → 12 sigmoid neurons
                         │
                         ▼
             12 → 3 output scores (softmax)
                         │
                         ▼
              clear / rain / storm
```

Each pixel is a normalized rain-intensity value. The toy generator draws a moving rain patch, then assigns a class according to the next map's average intensity. The perceptron uses the same history to solve the simpler binary question: will the next map contain rain?

## Important dimensions

For a batch of `B` examples, `X` has shape `(B, 48)`. The first weight matrix has shape `(48, 12)`, so `X @ W1` produces `(B, 12)`. The output matrix has shape `(12, 3)`, producing `(B, 3)` scores. Biases are `(1, 12)` and `(1, 3)` and broadcast across examples.

## Learning notes

- **Parameters:** the learned weight matrices and bias vectors.
- **Hyperparameters:** learning rate, hidden-layer width, epoch count, and batch size.
- **Activation:** sigmoid in the hidden layer; softmax converts output scores into class probabilities.
- **Loss:** multiclass cross-entropy measures the gap between predicted probabilities and correct labels.
- **Training:** forward pass → loss → backpropagated gradients → mini-batch parameter update.
- **Simplifying assumptions:** each example is a fixed-size, flattened map history; pixels are independent inputs to a dense network; the generated patterns are simple and deterministic in structure. This model does not represent physical atmospheric dynamics.

## Reference and attribution

Project inspiration: [Deep Learning for Short-Range Weather Forecasts](https://github.com/IAMIQBAL/Deep-Learning-for-Short-Range-Weather-Forecasts), which uses gridded weather variables and ConvLSTM models. This educational implementation is an independent, smaller NumPy demonstration; it does not copy the reference repository's source code or datasets.
