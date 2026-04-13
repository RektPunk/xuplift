use std::sync::Arc;

use faer::{Col, Mat};

use crate::feature_map::KernelFeatureMap;
use crate::xmodels::classifier::Classifier;
use crate::xmodels::regressor::Regressor;

/// R-Learner for Uplift Modeling.
///
/// Based on Robinson's transformation, this learner focuses on the residual-on-residual
/// regression to directly estimate the Heterogeneous Treatment Effect (HTE).
pub struct RLearner {
    /// Outcome nuisance model mu(x) = E[Y|X]
    pub mu: Regressor,

    /// Propensity nuisance model p(x) = E[T|X]
    pub p: Classifier,

    /// Treatment effect model
    pub tau: Regressor,
}

impl RLearner {
    /// Initializes and fits the RLearner using the provided data.
    ///
    /// The training process follows these steps:
    /// 1. Train m(x) to predict Y from X.
    /// 2. Train p(x) to predict T from X (Propensity).
    /// 3. Compute residuals: Y_tilde = Y - m(x) and T_tilde = T - p(x).
    /// 4. Train tau(x) by regressing Y_tilde on T_tilde.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix (n_samples x n_features).
    /// * `t` - The treatment assignment vector (n_samples, 0 or 1).
    /// * `y` - The observed outcome vector.
    pub fn new(x: &Mat<f32>, t: &Col<f32>, y: &Col<f32>) -> Self {
        let num_rows = x.nrows();

        // Train mu(x): Outcome model (E[Y|X])
        let mut mu_map = KernelFeatureMap::new();
        mu_map.fit(x);
        let mu_map_arc = Arc::new(mu_map);
        let mut mu = Regressor::new(Arc::clone(&mu_map_arc));
        mu.fit(y);

        // Train p(x): Propensity model (E[T|X])
        let mut p_map = KernelFeatureMap::new();
        p_map.fit(x);
        let p_map_arc = Arc::new(p_map);
        let mut p = Classifier::new(p_map_arc);
        p.fit(t, 20); // 20 iterations for classifier

        // Compute Residuals
        let mu_pred = mu.predict(x);
        let p_pred = p.predict(x);

        let mut y_tilde = Col::<f32>::zeros(num_rows);
        let mut t_tilde = Col::<f32>::zeros(num_rows);
        let mut r_target = Col::<f32>::zeros(num_rows);

        for i in 0..num_rows {
            y_tilde[i] = y[i] - mu_pred[i];
            t_tilde[i] = t[i] - p_pred[i].clamp(0.01, 0.99);

            // Objective: Minimize (y_tilde - t_tilde * tau)^2
            r_target[i] = y_tilde[i] / t_tilde[i];
        }

        // Train the final tau model on the R-objective target
        let mut tau_map = KernelFeatureMap::new();
        tau_map.fit(x);
        let mut tau = Regressor::new(Arc::new(tau_map));
        tau.fit(&r_target);

        Self { mu, p, tau }
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
