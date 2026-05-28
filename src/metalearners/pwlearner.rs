use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::classifier::Classifier;
use crate::xmodels::regressor::Regressor;

/// Propensity Score Weighted Learner (PW-Learner / IPW-Learner) for Uplift Modeling.
///
/// This learner uses inverse probability weighting (IPW) to transform the target variable,
/// correcting for confounding bias using only a propensity score model.
pub struct PWLearner {
    /// Treatment effect model trained on inverse-probability-weighted pseudo-outcomes
    pub tau: Regressor,
}

impl PWLearner {
    /// Initializes and fits the PWLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix (n_samples x n_features).
    /// * `t` - The treatment assignment vector (n_samples, 0 or 1).
    /// * `y` - The observed outcome vector.
    /// * `p_penalty` - The regularization penalty for the propensity classifier.
    /// * `p_max_iter` - The maximum number of iterations for the propensity classifier.
    /// * `tau_penalty` - The regularization penalty for the treatment effect model.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        p_penalty: f32,
        p_max_iter: usize,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Fit and predict propensity score model
        let mut p = Classifier::new(p_penalty, p_max_iter);
        p.fit(x, t);
        let p_pred = p.predict(x);

        // Construct Inverse Probability Weighted Pseudo-Outcomes
        // Y* = Y * [T / e(x) - (1-T) / (1-e(x))]
        let y_star = Col::<f32>::from_fn(num_rows, |i| {
            let gi = p_pred[i].clamp(0.01, 0.99); // Prevent division by zero
            if t[i] > 0.5 {
                y[i] / gi
            } else {
                -y[i] / (1.0 - gi)
            }
        });

        // Train the single tau model on the IPW target
        let mut tau = Regressor::new(tau_penalty);
        tau.fit(x, y_star.as_ref());

        Self { tau }
    }

    /// Estimates the uplift score: $\hat{\tau}(x)$ directly using the single PW model.
    pub fn predict_uplift(&self, x: MatRef<'_, f32>) -> Col<f32> {
        self.tau.predict(x)
    }

    /// Explains the uplift by decomposing the feature contributions of the single model.
    pub fn explain_uplift(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        self.tau.explain(x)
    }
}
