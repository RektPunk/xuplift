use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::regressor::Regressor;

/// Generalized R-Learner (GR-Learner) for Uplift Modeling.
///
/// This learner isolates the causal effect by residualizing both the outcome and the treatment.
/// Unlike the standard R-Learner which uses a Classifier for binary treatment propensity,
/// GR-Learner uses a Regressor for the treatment model, making it capable of handling
/// both continuous and binary treatment variables natively.
///
/// # Reference
/// * Nie, X., & Wager, S. (2021). Quasi-oracle estimation of heterogeneous treatment effects. Biometrika, 108(2), 299–319. https://doi.org/10.1093/biomet/asaa076
pub struct GRLearner {
    /// Treatment effect model
    pub tau: Regressor,
}

impl GRLearner {
    /// Initializes and fits the GRLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features).
    /// * `t` - Treatment vector (n_samples, continuous or binary).
    /// * `y` - Outcome vector (n_samples).
    /// * `is_categorical` - Vector indicating whether each feature is categorical (n_features).
    /// * `mu_penalty` - Regularization penalty for the outcome model.
    /// * `p_penalty` - Regularization penalty for the treatment model.
    /// * `tau_penalty` - Regularization penalty for the treatment effect model.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        is_categorical: &Vec<bool>,
        mu_penalty: f32,
        p_penalty: f32,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Fit and predict outcome model mu(x) and treatment model p(x) concurrently
        let (mu_pred, p_pred) = rayon::join(
            || {
                let mut mu = Regressor::new(mu_penalty);
                mu.fit(x, y, is_categorical);
                mu.predict(x)
            },
            || {
                let mut p = Regressor::new(p_penalty);
                p.fit(x, t, is_categorical);
                p.predict(x)
            },
        );

        // Compute residuals and weighted targets
        // Objective: Minimize (y_tilde - t_tilde * tau)^2
        // Equivalent to Weighted Least Squares: Minimize sum( w_i * (target_i - tau)^2 )
        let r_target_col = Col::<f32>::from_fn(num_rows, |i| {
            let y_tilde = y[i] - mu_pred[i];
            let t_tilde = t[i] - p_pred[i];
            if t_tilde.abs() > 1e-6 {
                y_tilde / t_tilde
            } else {
                0.0
            }
        });

        let r_weights_col = Col::<f32>::from_fn(num_rows, |i| {
            let t_tilde = t[i] - p_pred[i];
            t_tilde * t_tilde
        });

        // Fit model on weighted targets
        let mut tau = Regressor::new(tau_penalty);
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
