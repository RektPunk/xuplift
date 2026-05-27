use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::classifier::Classifier;
use crate::xmodels::regressor::Regressor;

/// X-Learner for Uplift Modeling.
///
/// This learner uses a three-stage process:
/// 1. Train outcome models $\mu_1(x)$ and $\mu_0(x)$.
/// 2. Impute treatment effects:
///    $D_1 = Y_1 - \mu_0(X_1)$
///    $D_0 = \mu_1(X_0) - Y_0$
/// 3. Train models $\tau_1(x)$ and $\tau_0(x)$ to predict $D_1$ and $D_0$.
///
/// The uplift is the propensity-weighted average:
/// $\tau(x) = g(x) \tau_0(x) + (1 - g(x)) \tau_1(x)$
/// where $g(x)$ is the propensity score $E[T|X]$.
pub struct XLearner {
    /// Imputed effect models
    pub tau_t1: Regressor,
    pub tau_t0: Regressor,

    /// Propensity model to weight the uplift estimates
    pub p: Classifier,
}

impl XLearner {
    /// Initializes and fits the XLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix (n_samples x n_features).
    /// * `t` - The treatment assignment vector (n_samples, 0 or 1).
    /// * `y` - The observed outcome vector.
    /// * `mu_penalty` - The regularization penalty for the regressor.
    /// * `p_penalty` - The regularization penalty for the propensity model.
    /// * `p_max_iter` - The maximum number of iterations for the propensity model.
    /// * `tau_penalty` - The regularization penalty for the tau models.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        mu_penalty: f32,
        p_penalty: f32,
        p_max_iter: usize,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Create weights for T=1 and T=0
        let w_t1 = Col::<f32>::from_fn(num_rows, |i| if t[i] > 0.5 { 1.0 } else { 0.0 });
        let w_t0 = Col::<f32>::from_fn(num_rows, |i| if t[i] <= 0.5 { 1.0 } else { 0.0 });

        // Train outcome models using weighted fitting on the full matrix
        let (mu_1, mu_0) = rayon::join(
            || {
                let mut mu_1 = Regressor::new(mu_penalty);
                mu_1.fit_weighted(x, y, &w_t1);
                mu_1
            },
            || {
                let mut mu_0 = Regressor::new(mu_penalty);
                mu_0.fit_weighted(x, y, &w_t0);
                mu_0
            },
        );

        // Impute Treatment Effects: D1 = Y - mu_0(X), D0 = mu_1(X) - Y
        // Compute these for all rows, but they will be filtered by weights during tau model fitting
        let (d_1, d_0) = rayon::join(|| y - mu_0.predict(x), || mu_1.predict(x) - y);

        // Train tau models and propensity model
        let ((tau_t1, tau_t0), p) = rayon::join(
            || {
                rayon::join(
                    || {
                        let mut tau_t1 = Regressor::new(tau_penalty);
                        tau_t1.fit_weighted(x, d_1.as_ref(), &w_t1);
                        tau_t1
                    },
                    || {
                        let mut tau_t0 = Regressor::new(tau_penalty);
                        tau_t0.fit_weighted(x, d_0.as_ref(), &w_t0);
                        tau_t0
                    },
                )
            },
            || {
                let mut p = Classifier::new(p_penalty, p_max_iter);
                p.fit(x, t);
                p
            },
        );

        Self { tau_t1, tau_t0, p }
    }

    /// Estimates the uplift score: $\tau(x) = g(x)\hat{\tau}_0(x) + (1 - g(x))\hat{\tau}_1(x)$
    pub fn predict_uplift(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let (g, (t_1, t_0)) = rayon::join(
            || self.p.predict(x), // P(T=1 | X)
            || rayon::join(|| self.tau_t1.predict(x), || self.tau_t0.predict(x)),
        );

        Col::from_fn(x.nrows(), |i| {
            let gi = g[i].clamp(0.01, 0.99);
            gi * t_0[i] + (1.0 - gi) * t_1[i]
        })
    }

    /// Explains the uplift by decomposing the weighted feature contributions.
    ///
    /// This method calculates the "Weighted Incremental Contribution" of each feature.
    /// Since the X-Learner prediction is a weighted sum of two tau models,
    /// the explanation is similarly derived by blending the feature-level contributions of $\tau_t1$ and $\tau_t0$:
    /// $Exp(x) = g(x) \cdot Exp_{\tau_t0}(x) + (1 - g(x)) \cdot Exp_{\tau_t1}(x)$
    ///
    /// # Returns
    /// A matrix (n_samples x n_features) representing how much each feature contributes to the final uplift score for each sample.
    pub fn explain_uplift(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        let (g, (exp_t1, exp_t0)) = rayon::join(
            || self.p.predict(x), // P(T=1 | X)
            || rayon::join(|| self.tau_t1.explain(x), || self.tau_t0.explain(x)),
        );

        Mat::from_fn(x.nrows(), x.ncols(), |i, j| {
            let gi = g[i].clamp(0.01, 0.99);
            gi * exp_t0[(i, j)] + (1.0 - gi) * exp_t1[(i, j)]
        })
    }
}
