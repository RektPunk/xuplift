use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::regressor::Regressor;

/// T-Learner (Two-Learner) for Uplift Modeling.
///
/// This learner splits the data by treatment assignment and trains two independent models:
/// $$\mu_1(x) = E[Y | X=x, T=1]$$
/// $$\mu_0(x) = E[Y | X=x, T=0]$$
/// The uplift is estimated as:
/// $$\tau(x) = \mu_1(x) - \mu_0(x)$$
///
/// # Reference
/// * Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). Metalearners for estimating heterogeneous treatment effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 4156–4165. https://doi.org/10.1073/pnas.1804597116
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
    /// * `x` - Feature matrix (n_samples x n_features).
    /// * `t` - Treatment vector (n_samples).
    /// * `y` - Outcome vector (n_samples).
    /// * `mu_penalty` - Regularization penalty for the outcome models.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        mu_penalty: f32,
    ) -> Self {
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

    /// Estimates the uplift score $\tau(x)$ for the given features.
    pub fn predict_uplift(&self, x: MatRef<'_, f32>) -> Col<f32> {
        let (pred_t1, pred_t0) = rayon::join(|| self.mu_t1.predict(x), || self.mu_t0.predict(x));
        pred_t1 - pred_t0
    }

    /// Explains the uplift by decomposing the feature contributions.
    pub fn explain_uplift(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        let (exp_t1, exp_t0) = rayon::join(|| self.mu_t1.explain(x), || self.mu_t0.explain(x));
        exp_t1 - exp_t0
    }
}
