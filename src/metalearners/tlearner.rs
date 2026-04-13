use std::sync::Arc;

use faer::{Col, Mat};

use crate::feature_map::KernelFeatureMap;
use crate::metalearners::data_utils;
use crate::xmodels::regressor::Regressor;

/// T-Learner (Two-Learner) for Uplift Modeling using Kernel-based Regressors.
///
/// This learner builds two independent models: one for the treatment group (T=1)
/// and one for the control group (T=0). The uplift is estimated as the difference
/// between the predictions of these two models.
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
    pub fn new(x: &Mat<f32>, t: &Col<f32>, y: &Col<f32>) -> Self {
        let num_rows = x.nrows();

        // Identify indices for T=1 and T=0
        let indices_t1: Vec<usize> = (0..num_rows).filter(|&i| t[i] > 0.5).collect();
        let indices_t0: Vec<usize> = (0..num_rows).filter(|&i| t[i] <= 0.5).collect();

        // Create sub-matrices
        let x_t1 = data_utils::filter_rows(x, &indices_t1);
        let y_t1 = data_utils::filter_elements(y, &indices_t1);

        let x_t0 = data_utils::filter_rows(x, &indices_t0);
        let y_t0 = data_utils::filter_elements(y, &indices_t0);

        // Train Model for T=1
        let mut map_t1 = KernelFeatureMap::new();
        map_t1.fit(&x_t1);
        let map_t1_arc = Arc::new(map_t1);
        let mut mu_t1 = Regressor::new(map_t1_arc);
        mu_t1.fit(&y_t1);

        // Train Model for T=0
        let mut map_t0 = KernelFeatureMap::new();
        map_t0.fit(&x_t0);
        let map_t0_arc = Arc::new(map_t0);
        let mut mu_t0 = Regressor::new(map_t0_arc);
        mu_t0.fit(&y_t0);

        Self { mu_t1, mu_t0 }
    }

    /// Estimates the uplift score: $\tau(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)$
    pub fn predict_uplift(&self, x: &Mat<f32>) -> Col<f32> {
        let pred_t1 = self.mu_t1.predict(x);
        let pred_t0 = self.mu_t0.predict(x);
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
        let exp_t1 = self.mu_t1.explain(x);
        let exp_t0 = self.mu_t0.explain(x);
        exp_t1 - exp_t0
    }
}
