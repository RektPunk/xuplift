use faer::{Col, Mat};
use rayon::prelude::*;

use crate::xmodels::classifier::Classifier;
use crate::xmodels::regressor::Regressor;

/// R-Learner for Uplift Modeling.
///
/// This learner focuses on the residual-on-residual regression.
/// It first trains an outcome model $m(x) = E[Y|X]$ and a propensity model $e(x) = E[T|X]$.
/// The treatment effect $\tau(x)$ is then estimated by minimizing the R-objective:
/// $$\min_{\tau} \sum_{i=1}^n [ (y_i - m(x_i)) - (t_i - e(x_i)) \tau(x_i) ]^2$$
pub struct RLearner {
    /// Treatment effect model
    pub tau: Regressor,
}

impl RLearner {
    /// Initializes and fits the RLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix (n_samples x n_features).
    /// * `t` - The treatment assignment vector (n_samples, 0 or 1).
    /// * `y` - The observed outcome vector.
    /// * `mu_penalty` - The regularization penalty for the regressor.
    /// * `p_penalty` - The regularization penalty for the propensity model.
    /// * `p_max_iter` - The maximum number of iterations for the propensity model.
    /// * `tau_penalty` - The regularization penalty for the treatment effect model.
    pub fn new(
        x: &Mat<f32>,
        t: &Col<f32>,
        y: &Col<f32>,
        mu_penalty: f32,
        p_penalty: f32,
        p_max_iter: usize,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Train and predict outcome model mu(x) and propensity model p(x)
        let (mu_pred, p_pred) = rayon::join(
            || {
                let mut mu = Regressor::new(mu_penalty);
                mu.fit(x, y);
                mu.predict(x)
            },
            || {
                let mut p = Classifier::new(p_penalty, p_max_iter);
                p.fit(x, t);
                p.predict(x)
            },
        );

        // Compute Residuals
        let (r_target, r_weights): (Vec<f32>, Vec<f32>) = (0..num_rows)
            .into_par_iter()
            .map(|i| {
                let y_tilde = y[i] - mu_pred[i];
                let t_tilde = t[i] - p_pred[i].clamp(0.01, 0.99);

                // Objective: Minimize (y_tilde - t_tilde * tau)^2
                // Equivalent to Weighted Least Squares: Minimize sum( w_i * (target_i - tau)^2 )
                // where target_i = y_tilde / t_tilde and w_i = t_tilde^2
                let weight = t_tilde * t_tilde;
                let target = if t_tilde.abs() > 1e-6 {
                    y_tilde / t_tilde
                } else {
                    0.0
                };
                (target, weight)
            })
            .unzip();

        let r_target_col = Col::<f32>::from_fn(num_rows, |i| r_target[i]);
        let r_weights_col = Col::<f32>::from_fn(num_rows, |i| r_weights[i]);

        // Train the final tau model on the R-objective target with weights
        let mut tau = Regressor::new(tau_penalty);
        tau.fit_weighted(x, &r_target_col, &r_weights_col);

        Self { tau }
    }

    /// Estimates the uplift score: $\hat{\tau}(x) = \arg\min_{\tau} \sum [ (Y - m(x)) - (T - e(x)) \cdot \tau(x) ]^2$
    pub fn predict_uplift(&self, x: &Mat<f32>) -> Col<f32> {
        self.tau.predict(x)
    }

    /// Explains the uplift by decomposing the feature contributions of the tau model.
    ///
    /// This explanation reveals how each feature contributes to the *change* in outcome
    /// caused by the treatment, rather than the outcome itself.
    ///
    /// Because R-Learner isolates the treatment signal by subtracting baseline expectations ($m(x)$ and $e(x)$),
    /// the feature contributions here are uniquely focused on "Causal Interaction" rather than simple correlation.
    ///
    /// # Returns
    /// A matrix (n_samples x n_features) showing the attribution of each feature
    /// to the final estimated Treatment Effect.
    pub fn explain_uplift(&self, x: &Mat<f32>) -> Mat<f32> {
        self.tau.explain(x)
    }
}
