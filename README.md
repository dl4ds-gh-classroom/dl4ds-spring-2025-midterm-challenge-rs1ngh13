## Overview

This repository contains the code for the midterm challenge of the course DS542 Deep Learning for Data Science.

The challenge is in three parts:
1. **Part 1 -- Simple CNN:** Define a relatively simple CNN model and train it on the CIFAR-100 dataset to
    get a complete pipeline and establish baseline performance.
2. **Part 2 -- More Sophisticated CNN Models:** Use a more sophisticated model, including predefined models from torchvision
   to train and evaluate on CIFAR-100.
3. **Part 3 -- Transfer Learning from a Pretrained Model:** Pretrain a model, or use one of the pretrained models from torchvision, and
   fine-tune it on CIFAR-100. Try to beat the best benchmark performance on the leaderboard.

## Files in Repository

Below is a breakdown of the files included in this repository, grouped by purpose, with descriptions based on their function in the project:

---

### Report & Guidelines

- `assignment_guidelines/` — Contents from the original `README.md` file provided by instructors, transferred here.
- `image-1.png`, `image-2.png`, `image-3.png`, `image-4.png` — Images used in the Assignment Report section of the `README.md`.

---

### Model Scripts

#### Model 1: Simple CNN

- `model_1_epoch_test.py` — Model 1 with different epoch values to determine which one has the best performance.
- `model_1.py` — Simple CNN with the best-performing configuration (epoch value set to 50).

#### Model 2: ResNet-18 (No Pretrained Weights)

- `model_2_hyperparam.py` — Model 2 (ResNet-18 without pretrained weights) running grid search to find the best configuration.
- `model_2_best_param.py` — Model 2 using the best configuration determined from `model_2_hyperparam.py`.

#### Model 3: ResNet-18 (Pretrained)

- `model_3_hyperparam.py` — Model 3 (ResNet-18 pretrained) running grid search to find the best configuration.
- `model_3_best_param.py` — Model 3 using the best configuration determined from `model_3_hyperparam.py`.

---

### Evaluation & Submission Files

- `eval_cifar100.py` — Evaluation code for the clean CIFAR-100 test set.
- `eval_ood.py` — Evaluation code for the out-of-distribution (OOD) test set.
- `submission_ood_model_1.py` — OOD output for Model 1 that was uploaded to Kaggle.
- `submission_ood_model_2.py` — OOD output for Model 2 that was uploaded to Kaggle.
- `submission_ood_model_3.py` — OOD output for Model 3 that was uploaded to Kaggle.
- `sample_submission.csv` — Sample output file for the OOD evaluation.

---

### Utility & Dependencies

- `utils.py` — Utility functions used in the training and evaluation pipeline.
- `requirements.txt` — Requirements needed to run scripts in this repository.

---

### Model Checkpoints

- `best_model.pth` — Saved PyTorch model from the best configuration.

---

### Starter Material

- `starter_code.py` — Starter code provided by instructors for the model framework.