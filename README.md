<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&height=300&color=gradient&text=xuplift&section=header&reversal=false&height=120&fontSize=90&fontColor=ff5500">
</div>

**xuplift** is a library for explainable uplift modeling. It uses linearized kernel feature maps to estimate treatment effects with both speed and mathematical rigor. Instead of computing a massive $N \times N$ kernel matrix, `xuplift` selects landmark points to project data into a finite-dimensional feature space.

## Supported Models
- Regressor: Kernel-based Ridge regressor for outcome and residual modeling.
- Classifier: Kernel-based Logistic classifier for precise propensity score estimation.

## Supported Meta-Learners
- DRClassifier, DRRegressor: Doubly robust estimator combining propensity scores and outcome models.
- GRClassifier, GRRegressor: Generalized R-learner supporting both continuous and binary treatments.
- MRegressor: Modified covariates learner optimized for randomized controlled trials (RCT).
- PWRegressor: Propensity score weighted learner using inverse probability weighting.
- RClassifier, RRegressor: Residual learner minimizing an R-objective via residual-on-residual regression.
- SClassifier, SRegressor: Single learner treating treatment assignment as a standard feature.
- TClassifier, TRegressor: Two learner approach fitting independent models for each group.
- XClassifier, XRegressor: Cross learner optimized for significantly unbalanced treatment groups.

# Installation
```bash
pip install xuplift
```
