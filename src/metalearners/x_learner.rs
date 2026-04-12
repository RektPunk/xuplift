use std::sync::Arc;

use faer::{Col, Mat};

use crate::feature_map::KernelFeatureMap;
use crate::xmodels::classifier::Classifier;
use crate::xmodels::regressor::Regressor;

/// X-Learner for Uplift Modeling.
///
/// This meta-learner is designed to handle significant class imbalance between
/// treatment and control groups by cross-predicting counterfactuals.
pub struct XLearner {
    /// Outcome models (Stage 1)
    pub mu_1: Regressor,
    pub mu_0: Regressor,

    /// Imputed effect models (Stage 2)
    pub tau_1: Regressor,
    pub tau_0: Regressor,

    /// Propensity model to weight the uplift estimates (Stage 3)
    pub p: Classifier,
}

impl XLearner {
    /// Initializes and fits the XLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix (n_samples x n_features).
    /// * `t` - The treatment assignment vector (n_samples, 0 or 1).
    /// * `y` - The observed outcome vector.
    pub fn new(x: &Mat<f32>, t: &Col<f32>, y: &Col<f32>) -> Self {
        let num_rows = x.nrows();

        // Identify indices for T=1 and T=0
        let indices_t1: Vec<usize> = (0..num_rows).filter(|&i| t[i] > 0.5).collect();
        let indices_t0: Vec<usize> = (0..num_rows).filter(|&i| t[i] <= 0.5).collect();

        // Create sub-matrices
        let x_t1 = Self::filter_rows(x, &indices_t1);
        let y_t1 = Self::filter_cols_vec(y, &indices_t1);
        let x_t0 = Self::filter_rows(x, &indices_t0);
        let y_t0 = Self::filter_cols_vec(y, &indices_t0);

        // Train Model for T=1
        let mut map_t1 = KernelFeatureMap::new();
        map_t1.fit(&x_t1);
        let map_t1_arc = Arc::new(map_t1);
        let mut mu_1 = Regressor::new(Arc::clone(&map_t1_arc));
        mu_1.fit(&y_t1);

        // Train Model for T=0
        let mut map_t0 = KernelFeatureMap::new();
        map_t0.fit(&x_t0);
        let map_t0_arc = Arc::new(map_t0);
        let mut mu_0 = Regressor::new(Arc::clone(&map_t0_arc));
        mu_0.fit(&y_t0);

        // Impute Treatment Effects and Train tau models
        // D_1 = Y_t1 - mu_0(X_t1): Actual treated outcome minus predicted control outcome (if they hadn't been treated)
        let d_1 = &y_t1 - &mu_0.predict(&x_t1);

        // D_0 = mu_1(X_t0) - Y_t0: Predicted treated outcome (if they had been treated) minus actual control outcome
        let d_0 = &mu_1.predict(&x_t0) - &y_t0;

        // Train tau models to estimate these imputed treatment effects (D_1, D_0)
        let mut tau_1 = Regressor::new(map_t1_arc);
        tau_1.fit(&d_1);
        let mut tau_0 = Regressor::new(map_t0_arc);
        tau_0.fit(&d_0);

        // Train Propensity Model (g): Predict T given X
        let mut propensity_map = KernelFeatureMap::new();
        propensity_map.fit(x);
        let propensity_map_arc = Arc::new(propensity_map);
        let mut p = Classifier::new(propensity_map_arc);
        p.fit(t, 20);

        Self {
            mu_1,
            mu_0,
            tau_1,
            tau_0,
            p,
        }
    }

    /// Predicts the Individual Treatment Effect (ITE / Uplift).
    pub fn predict_uplift(&self, x: &Mat<f32>) -> Col<f32> {
        let g = self.p.predict(x); // P(T=1 | X)
        let t_1 = self.tau_1.predict(x);
        let t_0 = self.tau_0.predict(x);

        let mut uplift = Col::<f32>::zeros(x.nrows());
        for i in 0..x.nrows() {
            let gi = g[i].clamp(0.01, 0.99);
            uplift[i] = gi * t_0[i] + (1.0 - gi) * t_1[i];
        }
        uplift
    }

    pub fn explain_uplift(&self, x: &Mat<f32>) -> Mat<f32> {
        let g = self.p.predict(x); // P(T=1 | X)

        let exp_t1 = self.tau_1.explain(x);
        let exp_t0 = self.tau_0.explain(x);

        let n_rows = x.nrows();
        let n_cols = x.ncols();

        Mat::from_fn(n_rows, n_cols, |i, j| {
            let gi = g[i].clamp(0.01, 0.99);
            gi * exp_t0[(i, j)] + (1.0 - gi) * exp_t1[(i, j)]
        })
    }

    fn filter_rows(x: &Mat<f32>, indices: &[usize]) -> Mat<f32> {
        let mut filtered = Mat::<f32>::zeros(indices.len(), x.ncols());
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            filtered
                .as_mut()
                .row_mut(new_idx)
                .copy_from(x.as_ref().row(old_idx));
        }
        filtered
    }

    fn filter_cols_vec(y: &Col<f32>, indices: &[usize]) -> Col<f32> {
        let mut filtered = Col::<f32>::zeros(indices.len());
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            filtered[new_idx] = y[old_idx];
        }
        filtered
    }
}
