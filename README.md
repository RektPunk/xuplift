<div style="text-align: center;">
  <img src="https://capsule-render.vercel.app/api?type=transparent&height=300&color=gradient&text=xuplift&section=header&reversal=false&height=120&fontSize=90&fontColor=ff5500">
</div>

**xuplift** is a library for explainable uplift modeling. It uses linearized kernel feature maps to estimate treatment effects with both speed and mathematical rigor. Instead of computing a massive $N \times N$ kernel matrix, `xuplift` selects landmark points to project data into a finite-dimensional feature space.

## Supported Models
- Regressor: Kernel-based Ridge regressor for outcome and residual modeling.
- Classifier: Kernel-based Logistic classifier for precise propensity score estimation.

## Supported Meta-Learners
- DRLearner: Doubly robust estimator combining propensity scores and outcome models.
- GRLearner: Generalized R-learner supporting both continuous and binary treatments.
- MLearner: Modified covariates learner optimized for randomized controlled trials (RCT).
- PWLearner: Propensity score weighted learner using inverse probability weighting.
- RLearner: Residual learner minimizing an R-objective via residual-on-residual regression.
- SLearner: Single learner treating treatment assignment as a standard feature.
- TLearner: Two learner approach fitting independent models for each group.
- XLearner: Cross learner optimized for significantly unbalanced treatment groups.

# Installation
```bash
pip install xuplift
```
