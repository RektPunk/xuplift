use std::sync::Arc;

use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::classifier::Classifier;
use crate::xmodels::feature_map::KernelFeatureMap;
use crate::xmodels::regressor::Regressor;

/// Doubly Robust (DR) Learner for Uplift Modeling.
///
/// This learner uses a multi-stage process with a doubly robust score wrapper:
/// 1. Fit baseline outcome models $\mu_1(x)$ and $\mu_0(x)$ on treatment/control groups.
/// 2. Fit a propensity model $e(x) = E[T|X]$ to estimate treatment assignment probabilities.
/// 3. Construct a doubly robust pseudo-outcome ($Y_{dr}$):
///    $$Y_{dr} = \mu_1(x) - \mu_0(x) + \frac{T(Y - \mu_1(x))}{e(x)} - \frac{(1 - T)(Y - \mu_0(x))}{1 - e(x)}$$
/// 4. Fit a model $\tau(x)$ on the full feature matrix to predict $Y_{dr}$.
///
/// # Reference
/// * Kennedy, E. H. (2023). Towards optimal doubly robust estimation of heterogeneous causal effects. Electronic Journal of Statistics, 17(2), 3008–3049. https://doi.org/10.1214/23-EJS2157
pub struct DRLearner {
    /// Treatment effect model
    pub tau: Regressor,
}

impl DRLearner {
    /// Initializes and fits the DRLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features).
    /// * `t` - Treatment vector (n_samples).
    /// * `y` - Outcome vector (n_samples).
    /// * `is_categorical` - Vector indicating whether each feature is categorical (n_features).
    /// * `max_bases` - Maximum number of landmark points for the kernel feature map.
    /// * `mu_penalty` - Regularization penalty for the outcome models.
    /// * `p_penalty` - Regularization penalty for the propensity model.
    /// * `p_max_iter` - Maximum iterations for the propensity model solver.
    /// * `tau_penalty` - Regularization penalty for the treatment effect model.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        is_categorical: &[bool],
        max_bases: usize,
        mu_penalty: f32,
        p_penalty: f32,
        p_max_iter: usize,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Fit KernelFeatureMap once and share it
        let mut map = KernelFeatureMap::new(max_bases);
        map.fit(x, is_categorical);
        let shared_map = Arc::new(map);

        // Pre-compute Z matrix once
        let z = shared_map.transform(x);
        let z_ref = z.as_ref();

        // Create weights for T=1 and T=0
        let w_t1 = Col::<f32>::from_fn(num_rows, |i| if t[i] > 0.5 { 1.0 } else { 0.0 });
        let w_t0 = Col::<f32>::from_fn(num_rows, |i| if t[i] <= 0.5 { 1.0 } else { 0.0 });

        // Fit and predict base outcome models and propensity model concurrently
        let ((mu_t1_pred, mu_t0_pred), p_pred) = rayon::join(
            || {
                rayon::join(
                    || {
                        let mut mu_t1 = Regressor::new(max_bases, mu_penalty);
                        mu_t1.kernel_feature_map = Some(shared_map.clone());
                        mu_t1.fit_with_z(z_ref, y, &w_t1);
                        mu_t1.predict_with_z(z_ref)
                    },
                    || {
                        let mut mu_t0 = Regressor::new(max_bases, mu_penalty);
                        mu_t0.kernel_feature_map = Some(shared_map.clone());
                        mu_t0.fit_with_z(z_ref, y, &w_t0);
                        mu_t0.predict_with_z(z_ref)
                    },
                )
            },
            || {
                let mut p = Classifier::new(max_bases, p_penalty, p_max_iter);
                p.kernel_feature_map = Some(shared_map.clone());
                p.fit_with_z(z_ref, t);
                p.predict_with_z(z_ref)
            },
        );

        // Construct pseudo-outcomes
        let y_dr = Col::<f32>::from_fn(num_rows, |i| {
            let gi = p_pred[i].clamp(0.01, 0.99);
            let mu_t1_i = mu_t1_pred[i];
            let mu_t0_i = mu_t0_pred[i];

            let base_effect = mu_t1_i - mu_t0_i;
            if t[i] > 0.5 {
                base_effect + (y[i] - mu_t1_i) / gi
            } else {
                base_effect - (y[i] - mu_t0_i) / (1.0 - gi)
            }
        });

        // Fit model on pseudo-outcomes
        let mut tau = Regressor::new(max_bases, tau_penalty);
        tau.kernel_feature_map = Some(shared_map.clone());
        let w_all = Col::<f32>::full(num_rows, 1.0);
        tau.fit_with_z(z_ref, y_dr.as_ref(), &w_all);

        Self { tau }
    }

    /// Estimates the uplift score $\tau(x)$ for the given features.
    pub fn predict_uplift(&self, x: MatRef<'_, f32>) -> Col<f32> {
        self.tau.predict(x)
    }

    /// Explains the uplift by decomposing the feature contributions.
    pub fn explain_uplift(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        self.tau.explain(x)
    }
}
