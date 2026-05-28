use crate::xmodels::regressor::Regressor;
use faer::{Col, ColRef, Mat, MatRef};

/// M-Learner (Modified Covariates Learner) for Uplift Modeling.
///
/// This learner transforms the target variable to isolate the causal effect in a single step,
/// assuming a randomized controlled trial (RCT) environment where the propensity score is 0.5.
///
/// # Reference
/// * Tian, L., Alizadeh, A. A., Gentles, A. J., & Tibshirani, R. (2014). A simple method for estimating interactions between a treatment and a large number of covariates. Journal of the American Statistical Association, 109(508), 1517–1532. https://doi.org/10.1080/01621459.2014.951443
pub struct MLearner {
    /// Treatment effect model
    pub tau: Regressor,
}

impl MLearner {
    /// Initializes and fits the MLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features).
    /// * `t` - Treatment vector (n_samples).
    /// * `y` - Outcome vector (n_samples).
    /// * `tau_penalty` - Regularization penalty for the treatment effect model.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Target Transformation: Y* = 2 * Y * (2T - 1)
        let y_star = Col::<f32>::from_fn(num_rows, |i| {
            let sign = if t[i] > 0.5 { 1.0 } else { -1.0 };
            2.0 * y[i] * sign
        });

        // Train the tau model on the modified target
        let mut tau = Regressor::new(tau_penalty);
        tau.fit(x, y_star.as_ref());

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
