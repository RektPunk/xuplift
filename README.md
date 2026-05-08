<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&height=300&color=gradient&text=xuplift&section=header&reversal=false&height=120&fontSize=90&fontColor=ff5500">
</div>
<p align="center">
  <a href="https://github.com/RektPunk/xuplift/releases/latest">
      <img alt="release" src="https://img.shields.io/github/v/release/RektPunk/xuplift.svg">
  </a>
  <a href="https://github.com/RektPunk/xuplift/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/RektPunk/xuplift.svg">
  </a>
</p>

Explainable uplift modeling via linearized kernel feature maps, providing a collection of meta-learners.

# Installation
Install using pip:
```bash
pip install xuplift
```

# Features
- Regressor: High-performance regression engine for outcome and residual modeling.
- Classifier: Optimized binary classifier for precise propensity score estimation.
- RLearner: Advanced residual-on-residual estimator with built-in 2-fold cross-fitting to ensure unbiased treatment effect estimation.
- XLearner: Optimized cross-learner designed to handle significantly unbalanced treatment groups.
- TLearner/SLearner: Standard two-model and single-model estimators for baseline causal analysis.
