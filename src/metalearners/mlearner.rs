use std::sync::Arc;

use faer::{Col, ColRef, Mat, MatRef};

use crate::xmodels::feature_map::KernelFeatureMap;
use crate::xmodels::regressor::Regressor;

/// Modified Covariates Regressor for Uplift Modeling.
///
/// This learner transforms the target variable to isolate the causal effect in one step,
/// assuming a randomized controlled trial (RCT) environment where the propensity score is 0.5.
///
/// # Reference
/// * Tian, L., Alizadeh, A. A., Gentles, A. J., & Tibshirani, R. (2014). A simple method for estimating interactions between a treatment and a large number of covariates. Journal of the American Statistical Association, 109(508), 1517–1532. https://doi.org/10.1080/01621459.2014.951443
pub struct MRegressor {
    /// Treatment effect model
    pub tau: Regressor,
}

impl MRegressor {
    /// Initializes and fits the MRegressor using the provided data.
    ///
    /// # Arguments
    /// * `x` - Feature matrix (n_samples x n_features).
    /// * `t` - Treatment vector (n_samples).
    /// * `y` - Outcome vector (n_samples).
    /// * `is_categorical` - Vector indicating whether each feature is categorical (n_features).
    /// * `max_bases` - Maximum number of landmark points for the kernel feature map.
    /// * `tau_penalty` - Regularization penalty for the treatment effect model.
    pub fn new(
        x: MatRef<'_, f32>,
        t: ColRef<'_, f32>,
        y: ColRef<'_, f32>,
        is_categorical: &[bool],
        max_bases: usize,
        tau_penalty: f32,
    ) -> Self {
        let num_rows = x.nrows();

        // Fit KernelFeatureMap once and share it
        let mut map = KernelFeatureMap::new(max_bases);
        map.fit(x, is_categorical);
        let shared_map = Arc::new(map);

        // Pre-compute Z matrix once
        let z = shared_map.transform(x);
        let z_ref = z.as_ref();

        // Target Transformation: Y* = 2 * Y * (2T - 1)
        let y_m = Col::from_fn(num_rows, |i| {
            let sign = if t[i] > 0.5 { 1.0 } else { -1.0 };
            2.0 * y[i] * sign
        });

        // Fit Regressor on pseudo-outcomes
        let mut tau = Regressor::new(max_bases, tau_penalty);
        tau.kernel_feature_map = Some(shared_map.clone());
        let w_all = Col::<f32>::full(num_rows, 1.0);
        tau.fit_with_z(z_ref, y_m.as_ref(), &w_all);

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
