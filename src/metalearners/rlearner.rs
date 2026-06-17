use std::sync::Arc;

use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::classifier::Classifier;
use crate::xmodels::feature_map::KernelFeatureMap;
use crate::xmodels::regressor::Regressor;

/// R-Learner (Residual Learner) for Uplift Modeling.
///
/// This learner focuses on the residual-on-residual regression.
/// It first fits an outcome model $m(x) = E[Y|X]$ and a propensity model $e(x) = E[T|X]$.
/// The treatment effect $\tau(x)$ is then estimated by minimizing the R-objective:
/// $$\min_{\tau} \sum_{i=1}^n [ (y_i - m(x_i)) - (t_i - e(x_i)) \tau(x_i) ]^2$$
///
/// # Reference
/// * Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076
pub struct RLearner {
    /// Treatment effect model
    pub tau: Regressor,
}

impl RLearner {
    /// Initializes and fits the RLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features).
    /// * `t` - Treatment vector (n_samples).
    /// * `y` - Outcome vector (n_samples).
    /// * `is_categorical` - Vector indicating whether each feature is categorical (n_features).
    /// * `max_bases` - Maximum number of bases for the kernel feature map.
    /// * `mu_penalty` - Regularization penalty for the outcome model.
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

        // Fit and predict outcome model mu(x) and propensity model p(x) concurrently
        let (mu_pred, p_pred) = rayon::join(
            || {
                let mut mu = Regressor::new(max_bases, mu_penalty);
                mu.kernel_feature_map = Some(shared_map.clone());
                mu.fit(x, y, is_categorical);
                mu.predict(x)
            },
            || {
                let mut p = Classifier::new(max_bases, p_penalty, p_max_iter);
                p.kernel_feature_map = Some(shared_map.clone());
                p.fit(x, t, is_categorical);
                p.predict(x)
            },
        );

        // Compute residuals and weighted targets
        // Objective: Minimize (y_tilde - t_tilde * tau)^2
        // Equivalent to Weighted Least Squares: Minimize sum( w_i * (target_i - tau)^2 )
        // where target_i = y_tilde / t_tilde and w_i = t_tilde^2
        let r_target_col = Col::<f32>::from_fn(num_rows, |i| {
            let y_tilde = y[i] - mu_pred[i];
            let t_tilde = t[i] - (p_pred[i].clamp(0.01, 0.99));

            if t_tilde.abs() > 1e-6 {
                y_tilde / t_tilde
            } else {
                0.0
            }
        });

        let r_weights_col = Col::<f32>::from_fn(num_rows, |i| {
            let t_tilde = t[i] - (p_pred[i].clamp(0.01, 0.99));
            t_tilde * t_tilde
        });

        // Fit model on weighted targets
        let mut tau = Regressor::new(max_bases, tau_penalty);
        tau.kernel_feature_map = Some(shared_map.clone());
        tau.fit_weighted(x, r_target_col.as_ref(), &r_weights_col, is_categorical);

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
