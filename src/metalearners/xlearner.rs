use std::sync::Arc;

use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::classifier::Classifier;
use crate::xmodels::feature_map::KernelFeatureMap;
use crate::xmodels::regressor::Regressor;

/// X-Learner (Cross-Learner) for Uplift Modeling.
///
/// This learner uses a three-stage process:
/// 1. Fit outcome models $\mu_1(x)$ and $\mu_0(x)$.
/// 2. Impute treatment effects:
///    $D_1 = Y_1 - \mu_0(X_1)$
///    $D_0 = \mu_1(X_0) - Y_0$
/// 3. Fit models $\tau_1(x)$ and $\tau_0(x)$ to predict $D_1$ and $D_0$.
///
/// The uplift is the propensity-weighted average:
/// $\tau(x) = g(x) \tau_0(x) + (1 - g(x)) \tau_1(x)$
/// where $g(x)$ is the propensity score $E[T|X]$.
///
/// # Reference
/// * Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116
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
    /// * `x` - Feature matrix (n_samples x n_features).
    /// * `t` - Treatment vector (n_samples).
    /// * `y` - Outcome vector (n_samples).
    /// * `is_categorical` - Vector indicating whether each feature is categorical (n_features).
    /// * `mu_penalty` - Regularization penalty for the outcome models.
    /// * `p_penalty` - Regularization penalty for the propensity model.
    /// * `p_max_iter` - Maximum iterations for the propensity model solver.
    /// * `tau_penalty` - Regularization penalty for the treatment effect models.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        is_categorical: &Vec<bool>,
        mu_penalty: f32,
        p_penalty: f32,
        p_max_iter: usize,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Fit KernelFeatureMap once and share it
        let mut map = KernelFeatureMap::new();
        map.fit(x, is_categorical);
        let shared_map = Arc::new(map);

        // Create weights for T=1 and T=0
        let w_t1 = Col::<f32>::from_fn(num_rows, |i| if t[i] > 0.5 { 1.0 } else { 0.0 });
        let w_t0 = Col::<f32>::from_fn(num_rows, |i| if t[i] <= 0.5 { 1.0 } else { 0.0 });

        // Compute residuals by using outcome models concurrently
        let (d_0, d_1) = rayon::join(
            || {
                let mut mu_t1 = Regressor::new(mu_penalty);
                mu_t1.kernel_feature_map = Some(shared_map.clone());
                mu_t1.fit_weighted(x, y, &w_t1, is_categorical);
                mu_t1.predict(x) - y
            },
            || {
                let mut mu_t0 = Regressor::new(mu_penalty);
                mu_t0.kernel_feature_map = Some(shared_map.clone());
                mu_t0.fit_weighted(x, y, &w_t0, is_categorical);
                y - mu_t0.predict(x)
            },
        );

        // Fit tau models and propensity model concurrently
        let ((tau_t1, tau_t0), p) = rayon::join(
            || {
                rayon::join(
                    || {
                        let mut tau_t1 = Regressor::new(tau_penalty);
                        tau_t1.kernel_feature_map = Some(shared_map.clone());
                        tau_t1.fit_weighted(x, d_1.as_ref(), &w_t1, is_categorical);
                        tau_t1
                    },
                    || {
                        let mut tau_t0 = Regressor::new(tau_penalty);
                        tau_t0.kernel_feature_map = Some(shared_map.clone());
                        tau_t0.fit_weighted(x, d_0.as_ref(), &w_t0, is_categorical);
                        tau_t0
                    },
                )
            },
            || {
                let mut p = Classifier::new(p_penalty, p_max_iter);
                p.kernel_feature_map = Some(shared_map.clone());
                p.fit(x, t, is_categorical);
                p
            },
        );

        Self { tau_t1, tau_t0, p }
    }

    /// Estimates the uplift score $\tau(x)$ for the given features.
    pub fn predict_uplift(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let (g, (tau_t1_pred, tau_t0_pred)) = rayon::join(
            || self.p.predict(x), // P(T=1 | X)
            || rayon::join(|| self.tau_t1.predict(x), || self.tau_t0.predict(x)),
        );

        Col::from_fn(x.nrows(), |i| {
            let gi = g[i].clamp(0.01, 0.99);
            gi * tau_t0_pred[i] + (1.0 - gi) * tau_t1_pred[i]
        })
    }

    /// Explains the uplift by decomposing the feature contributions.
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
