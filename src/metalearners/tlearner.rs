use faer::{Col, Mat};

use crate::xmodels::regressor::Regressor;

/// T-Learner (Two-Learner) for Uplift Modeling using Kernel-based Regressors.
///
/// This learner splits the data by treatment assignment and trains two independent models:
/// $$\mu_1(x) = E[Y | X=x, T=1]$$
/// $$\mu_0(x) = E[Y | X=x, T=0]$$
/// The uplift is estimated as:
/// $$\tau(x) = \mu_1(x) - \mu_0(x)$$
pub struct TLearner {
    /// Regressor trained exclusively on the treatment group (T=1).
    pub mu_t1: Regressor,
    /// Regressor trained exclusively on the control group (T=0).
    pub mu_t0: Regressor,
}

impl TLearner {
    /// Initializes and fits the TLearner by splitting the data into treatment and control groups.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix.
    /// * `t` - The treatment assignment vector (0 or 1).
    /// * `y` - The observed outcome vector.
    /// * `mu_penalty` - The regularization penalty for the regressor.
    pub fn new(x: &Mat<f32>, t: &Col<f32>, y: &Col<f32>, mu_penalty: f32) -> Self {
        let num_rows = x.nrows();

        // Create weights for T=1 and T=0
        let w_t1 = Col::<f32>::from_fn(num_rows, |i| if t[i] > 0.5 { 1.0 } else { 0.0 });
        let w_t0 = Col::<f32>::from_fn(num_rows, |i| if t[i] <= 0.5 { 1.0 } else { 0.0 });

        // Train Models using weighted fitting on the full original matrix
        let (mu_t1, mu_t0) = rayon::join(
            || {
                let mut mu_t1 = Regressor::new(mu_penalty);
                mu_t1.fit_weighted(x, y, &w_t1);
                mu_t1
            },
            || {
                let mut mu_t0 = Regressor::new(mu_penalty);
                mu_t0.fit_weighted(x, y, &w_t0);
                mu_t0
            },
        );

        Self { mu_t1, mu_t0 }
    }

    /// Estimates the uplift score: $\tau(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$
    pub fn predict_uplift(&self, x: &Mat<f32>) -> Col<f32> {
        let (pred_t1, pred_t0) = rayon::join(|| self.mu_t1.predict(x), || self.mu_t0.predict(x));
        pred_t1 - pred_t0
    }

    /// Explains the uplift by comparing feature contributions from both models.
    ///
    /// Since T-Learner uses two separate models, the uplift explanation is the
    /// difference between the feature importance/contribution of the T=1 model
    /// and the T=0 model.
    ///
    /// # Returns
    /// A matrix (n_samples x n_features) representing the incremental contribution of each feature.
    pub fn explain_uplift(&self, x: &Mat<f32>) -> Mat<f32> {
        let (exp_t1, exp_t0) = rayon::join(|| self.mu_t1.explain(x), || self.mu_t0.explain(x));
        exp_t1 - exp_t0
    }
}
