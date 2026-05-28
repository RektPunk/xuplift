use faer::{Col, ColRef, Mat, MatRef};
use rayon::prelude::*;

use crate::xmodels::regressor::Regressor;

/// Generalized R-Learner (GRLearner) for Uplift Modeling / Continuous Treatment.
///
/// This learner isolates the causal effect by residualizing both the outcome and the treatment.
/// Unlike the standard R-Learner which uses a Classifier for binary treatment propensity,
/// GRLearner uses a Regressor for the treatment model, making it capable of handling
/// both continuous and binary treatment variables natively.
pub struct GRLearner {
    /// Treatment effect model
    pub tau: Regressor,
}

impl GRLearner {
    /// Initializes and fits the GRLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix (n_samples x n_features).
    /// * `t` - The treatment vector (n_samples, continuous or binary).
    /// * `y` - The observed outcome vector.
    /// * `mu_penalty` - The regularization penalty for the outcome model.
    /// * `t_penalty` - The regularization penalty for the treatment model.
    /// * `tau_penalty` - The regularization penalty for the final treatment effect model.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        mu_penalty: f32,
        p_penalty: f32,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Train and predict outcome model mu(x) and treatment model m_t(x) in parallel
        // GRLearner uses Regressor for BOTH models to support continuous treatment.
        let (mu_pred, t_pred) = rayon::join(
            || {
                let mut mu = Regressor::new(mu_penalty);
                mu.fit(x, y);
                mu.predict(x)
            },
            || {
                let mut model_t = Regressor::new(p_penalty);
                model_t.fit(x, t);
                model_t.predict(x)
            },
        );

        // Compute Residuals & Construct Weighted Targets
        let (r_target, r_weights): (Vec<f32>, Vec<f32>) = (0..num_rows)
            .into_par_iter()
            .map(|i| {
                let y_tilde = y[i] - mu_pred[i];
                let t_tilde = t[i] - t_pred[i];

                // Objective: Minimize (y_tilde - t_tilde * tau)^2
                // Equivalent to Weighted Least Squares: Minimize sum( w_i * (target_i - tau)^2 )
                let weight = t_tilde * t_tilde;
                let target = if t_tilde.abs() > 1e-5 {
                    y_tilde / t_tilde
                } else {
                    0.0
                };
                (target, weight)
            })
            .unzip();

        let r_target_col = Col::<f32>::from_fn(num_rows, |i| r_target[i]);
        let r_weights_col = Col::<f32>::from_fn(num_rows, |i| r_weights[i]);

        // Train the tau model on the R-objective target with weights
        let mut tau = Regressor::new(tau_penalty);
        tau.fit_weighted(x, r_target_col.as_ref(), &r_weights_col);

        Self { tau }
    }

    /// Estimates the uplift score: $\hat{\tau}(x)$.
    pub fn predict_uplift(&self, x: MatRef<'_, f32>) -> Col<f32> {
        self.tau.predict(x)
    }

    /// Explains the uplift by decomposing the feature contributions of the tau model.
    pub fn explain_uplift(&self, x: MatRef<'_, f32>) -> Mat<f32> {
        self.tau.explain(x)
    }
}
