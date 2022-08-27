# Code implementation of Adversarial Security Verification of Data-driven FDC Systems
This repository contains the implementations of all adversarial attack and defense methods in the paper: 
> Adversarial Security Verification of Data-driven FDC Systems.
> 
> Yue Zhuo and Zhiqiang Ge*

We also released the trained model checkpoints and calculated adversarial samples.

## Repository Structure
`attack/` contains the implementations of adversarial attack methods

`defense/` contains the implementations of adversarial defense methods

`data/` contains the original TEP data with the normal condition and first 15 faults

`models/` contains the the trained model checkpoints of defense methods and calculated adversarial samples of attack methods

`TEP/` for fault classification

`TEP_FD/` for fault detection

## Run
Run `main.py` can iteratively train the defensive models and calculate adversarial samples, to reproduce the main results of benchmark in the paper.

Configure variable `dataset_name` to adjust the task (faule classification or detection).

## Code Reference
