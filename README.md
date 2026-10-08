# FashionMNIST CNN Depth Study

A controlled experimental study investigating how **CNN depth and width affect image-classification performance** on the FashionMNIST dataset.

The study compares three convolutional neural network architectures under the same training configuration and evaluates their performance across multiple random seeds. In addition to comparing mean validation accuracy, the project examines run-to-run variability and uses paired statistical tests to determine whether observed differences are likely to be meaningful.

---

## Research Question

**How does increasing convolutional network depth or width affect classification performance on FashionMNIST under a controlled training configuration?**

The experiment specifically compares:

* **CNN5** — baseline architecture with 5 convolutional layers
* **CNN5Wide** — wider version of the 5-layer architecture
* **CNN7** — deeper architecture with 7 convolutional layers

The goal is not to determine whether one architecture is universally superior, but to evaluate how architectural changes behave under the experimental conditions used in this study.

---

## Experimental Design

To make the comparison more reliable, the architectures were evaluated using the same general training protocol.

### Models

| Model    | Description            | Parameters |
| -------- | ---------------------- | ---------: |
| CNN5     | 5 convolutional layers |  2,142,474 |
| CNN5Wide | Wider 5-layer CNN      |  2,386,954 |
| CNN7     | 7 convolutional layers |  2,326,986 |

### Repeated Runs

Each architecture was trained using five random seeds:

```text
42, 43, 44, 45, 46
```

This resulted in:

**3 architectures × 5 seeds = 15 experimental runs**

Using multiple seeds reduces the risk of drawing conclusions from a single favorable or unfavorable training run.

---

## Dataset

The experiments use the **FashionMNIST** dataset.

FashionMNIST contains grayscale images belonging to 10 clothing-related categories.

### Classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset is commonly used as a benchmark for image-classification experiments and provides a useful environment for studying model architecture without requiring large computational resources.

---

## Training Configuration

The architectures were evaluated under a controlled training setup.

Key configuration included:

* **Framework:** PyTorch
* **Dataset:** FashionMNIST
* **Number of classes:** 10
* **Input:** Grayscale images
* **Optimizer:** Adam
* **Learning rate:** 0.001
* **Loss function:** Cross-Entropy Loss
* **Batch size:** 128
* **Maximum training epochs:** 10
* **Random seeds:** 42–46

The best checkpoint for each run was selected according to the **lowest validation loss**.

The corresponding validation accuracy is reported as the performance of that selected checkpoint.

---

## Results

### Mean Validation Performance

| Model    | Mean Validation Accuracy |  Std. Dev. |        Range | Mean Best Epoch |
| -------- | -----------------------: | ---------: | -----------: | --------------: |
| **CNN5** |               **93.29%** | **0.162%** | 93.02–93.43% |             5.0 |
| CNN5Wide |                   93.19% |     0.220% | 92.88–93.40% |             4.8 |
| CNN7     |                   92.55% |     0.328% | 92.10–92.87% |             5.2 |

### Main Observation

CNN5 achieved the highest mean validation accuracy while also showing the lowest run-to-run variability.

Despite having more parameters, CNN5Wide did not improve performance over CNN5.

CNN7 also used more parameters than CNN5 but produced substantially lower validation accuracy.

---

## Performance by Random Seed

### CNN5

| Seed | Validation Accuracy |
| ---: | ------------------: |
|   42 |              93.30% |
|   43 |              93.38% |
|   44 |              93.02% |
|   45 |              93.43% |
|   46 |              93.32% |

### CNN5Wide

| Seed | Validation Accuracy |
| ---: | ------------------: |
|   42 |              93.40% |
|   43 |              93.08% |
|   44 |              93.40% |
|   45 |              92.88% |
|   46 |              93.18% |

### CNN7

| Seed | Validation Accuracy |
| ---: | ------------------: |
|   42 |              92.68% |
|   43 |              92.32% |
|   44 |              92.78% |
|   45 |              92.87% |
|   46 |              92.10% |

---

## Statistical Analysis

Because the same random seeds were used for each architecture, paired comparisons were performed using a **paired t-test**.

### CNN5 vs CNN7

Mean performance difference:

**+0.74 percentage points in favor of CNN5**

Paired t-test:

```text
t = 4.1501
p = 0.0143
df = 4
```

At the conventional significance level of α = 0.05:

```text
p < 0.05
```

Therefore, the difference between CNN5 and CNN7 is statistically significant within this experiment.

### CNN5 vs CNN5Wide

Mean performance difference:

**+0.10 percentage points in favor of CNN5**

Paired t-test:

```text
t = 0.6218
p = 0.5677
df = 4
```

Since:

```text
p > 0.05
```

there is not sufficient evidence to conclude that CNN5 and CNN5Wide have different performance under this experimental setup.

---

## Interpretation

The results suggest that increasing convolutional depth from five to seven layers did **not** improve FashionMNIST validation performance under the controlled training configuration used in this study.

In fact, CNN7 achieved a lower mean validation accuracy than CNN5, and the paired statistical test indicates that this difference was statistically significant across the five matched random seeds.

Increasing the width of the five-layer architecture produced a different result. CNN5Wide had approximately 245,000 more parameters than CNN5, but its mean validation accuracy was almost identical. The paired test did not find a statistically significant difference.

These results suggest that simply increasing model capacity does not necessarily lead to better performance on this task.

---

## Training Behaviour

The training curves also show a gap between training and validation performance after the best validation checkpoint.

For example, in several runs, training accuracy continued increasing while validation accuracy stopped improving or slightly declined.

This pattern is consistent with increasing overfitting during later training epochs.

The use of validation-loss-based checkpoint selection helps prevent the final model from simply being the model obtained at the last training epoch.

---

## Key Findings

### 1. CNN5 achieved the best overall validation performance

CNN5 achieved:

**93.29% mean validation accuracy**

and the lowest standard deviation:

**0.162 percentage points**

among the three architectures.

### 2. Increasing depth did not improve performance

CNN7 achieved:

**92.55% mean validation accuracy**

which was approximately:

**0.74 percentage points lower than CNN5.**

The paired test produced:

**p = 0.0143**

indicating a statistically significant difference in this experiment.

### 3. Increasing width provided no meaningful improvement

CNN5Wide increased the parameter count from approximately 2.14M to 2.39M, but achieved:

**93.19% mean validation accuracy**

compared with 93.29% for CNN5.

The difference was not statistically significant:

**p = 0.5677**

### 4. More parameters did not guarantee better performance

Both CNN5Wide and CNN7 contained more parameters than CNN5, yet neither achieved better mean validation performance.

This demonstrates why model evaluation should consider both **performance and model complexity**, rather than assuming that larger models are automatically better.

---

## Limitations

This study should be interpreted within the scope of its experimental setup.

Important limitations include:

* Only three CNN architectures were evaluated.
* Five random seeds were used for each architecture.
* The experiments were conducted on a single dataset.
* The training configuration was fixed rather than extensively optimized for each architecture.
* The paired t-tests were performed with only five matched observations per comparison.
* Results may differ with other datasets, hyperparameters, optimization strategies, or architecture designs.

Therefore, the results should not be interpreted as evidence that five-layer CNNs are universally better than deeper networks.

Instead, the study demonstrates that **under the controlled conditions used here, increasing depth from five to seven convolutional layers did not improve performance and was associated with significantly lower validation accuracy.**

---

## Reproducibility

The project stores the experimental outputs so that the analysis can be reproduced without repeating the computationally expensive training experiments.

The main result files are:

```text
results/
├── raw_results.json
├── summary_statistics.csv
├── validation_accuracy_comparison.png
└── validation_accuracy_by_seed.png
```

The raw JSON file contains the results from all 15 experimental runs, including:

* model architecture
* random seed
* parameter count
* best epoch
* validation loss
* validation accuracy
* training loss
* training accuracy
* validation loss history
* validation accuracy history

---

## Project Structure

```text
fashionmnist-cnn-depth-study/
│
├── data/
│   └── FashionMNIST/
│
├── notebooks/
│   └── analysis.ipynb
│
├── results/
│   ├── raw_results.json
│   ├── summary_statistics.csv
│   ├── validation_accuracy_comparison.png
│   └── validation_accuracy_by_seed.png
│
├── src/
│   ├── analyze_results.py
│   ├── evaluate.py
│   ├── models.py
│   ├── run_experiment.py
│   ├── run_one_test.py
│   ├── run_sanity_check.py
│   ├── train.py
│   └── utils.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* SciPy

---

## Future Work

Several extensions could make the study more comprehensive:

* Evaluate additional CNN depths.
* Investigate different width configurations.
* Compare parameter efficiency more systematically.
* Increase the number of random seeds.
* Evaluate the architectures on additional image datasets.
* Study the effect of normalization and regularization techniques.
* Compare training efficiency and computational cost.
* Investigate whether architectural changes affect different FashionMNIST classes differently.

---

## Conclusion

This study investigated whether increasing CNN depth or width improves FashionMNIST classification performance under a controlled experimental configuration.

Among the three architectures, **CNN5 achieved the strongest overall validation performance and the lowest run-to-run variability**.

Increasing depth to seven convolutional layers resulted in lower validation performance, with the difference from CNN5 being statistically significant across the matched random seeds. In contrast, increasing the width of the five-layer network did not produce a statistically significant performance improvement despite increasing the number of parameters.

The results highlight an important principle in neural-network experimentation:

> **Increasing model capacity does not automatically produce better generalization.**

Careful experimental control, repeated runs, statistical comparison, and analysis of model complexity are necessary when evaluating architectural changes.
