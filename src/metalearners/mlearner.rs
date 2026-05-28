use crate::xmodels::regressor::Regressor;
use faer::{Col, ColRef, Mat, MatRef};

/// M-Learner (Modified Covariates / Interaction Learner) for Uplift Modeling.
///
/// This learner transforms the target variable to isolate the causal effect in a single step,
/// assuming a randomized controlled trial (RCT) environment where the propensity score is 0.5.
pub struct MLearner {
    /// Single regression model trained on the modified target
    pub tau: Regressor,
}

impl MLearner {
    /// Initializes and fits the MLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix (n_samples x n_features).
    /// * `t` - The treatment assignment vector (n_samples, 0 or 1).
    /// * `y` - The observed outcome vector.
    /// * `tau_penalty` - The regularization penalty for the single model.
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

        // Train a single tau model directly on the modified target
        let mut tau = Regressor::new(tau_penalty);
        tau.fit(x, y_star.as_ref());

        Self { tau }
    }

    /// Estimates the uplift score $\hat{\tau}(x)$ directly using the single modified model.
    pub fn predict_uplift(&self, x: MatRef<'_, f32>) -> Col<f32> {
        self.tau.predict(x)
    }

    /// Explains the uplift by decomposing the feature contributions of the single model.
    pub fn explain_uplift(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        self.tau.explain(x)
    }
}
