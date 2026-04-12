use faer::{Col, Mat};

use crate::feature_map::KernelFeatureMap;
use crate::xmodels::regressor::Regressor;

/// S-Learner (Single Learner) for Uplift Modeling using a Kernel-based Regressor.
///
/// This learner treats the treatment assignment as an additional feature in a single
/// response surface model. The Individual Treatment Effect (ITE) is estimated by
/// taking the difference between predictions with the treatment feature set to 1 and 0.
pub struct SLearner {
    /// The underlying Regressor trained on augmented features (X, T).
    pub regressor: Regressor,
}

impl SLearner {
    /// Initializes and fits the SLearner using the provided data.
    ///
    /// # Arguments
    /// * `x` - The original feature matrix (n_samples x n_features).
    /// * `t` - The treatment assignment vector (n_samples, 0 or 1).
    /// * `y` - The observed outcome vector.
    ///
    /// This method automatically handles feature augmentation, kernel mapping,
    /// and regression fitting.
    pub fn new(x: &Mat<f32>, t: &Col<f32>, y: &Col<f32>) -> Self {
        let num_rows = x.nrows();
        let num_cols = x.ncols();
        let mut x_combined = Mat::<f32>::zeros(num_rows, num_cols + 1);

        // Copy original features X into the augmented matrix
        x_combined
            .as_mut()
            .submatrix_mut(0, 0, num_rows, num_cols)
            .copy_from(x);

        // Append the treatment vector T as the final column
        for i in 0..num_rows {
            x_combined[(i, num_cols)] = t[i];
        }

        // Initialize and fit the KernelFeatureMap with the augmented dimension (k + 1)
        let mut feature_map = KernelFeatureMap::new();
        feature_map.fit(&x_combined);

        // Initialize and fit the Regressor using the generated kernel features
        let mut regressor = Regressor::new(feature_map);
        regressor.fit(y);

        Self { regressor }
    }

    /// Estimates the uplift score (Conditional Average Treatment Effect).
    ///
    /// It calculates: $\tau(x) = E[Y | X=x, T=1] - E[Y | X=x, T=0]$
    ///
    /// # Arguments
    /// * `x` - The feature matrix for which to estimate uplift.
    pub fn predict_uplift(&self, x: &Mat<f32>) -> Col<f32> {
        let num_rows = x.nrows();
        let num_cols = x.ncols();

        // Use a scratchpad to avoid redundant allocations for X
        let mut scratch = Mat::<f32>::zeros(num_rows, num_cols + 1);
        scratch
            .as_mut()
            .submatrix_mut(0, 0, num_rows, num_cols)
            .copy_from(x);

        // Case 1: Treatment = 1
        scratch.as_mut().col_mut(num_cols).fill(1.0);
        let pred_t1 = self.regressor.predict(&scratch);

        // Case 2: Treatment = 0
        scratch.as_mut().col_mut(num_cols).fill(0.0);
        let pred_t0 = self.regressor.predict(&scratch);

        pred_t1 - pred_t0
    }

    /// Explains the uplift by decomposing the difference in feature contributions.
    ///
    /// This method calculates the "Incremental Contribution" of each feature,
    /// showing how the treatment changes the impact of each variable on the outcome.
    ///
    /// # Returns
    /// A matrix of dimensions (n_samples x n_features + 1), where the last column
    /// represents the direct effect of the treatment variable itself.
    pub fn explain_uplift(&self, x: &Mat<f32>) -> Mat<f32> {
        let num_rows = x.nrows();
        let num_cols = x.ncols();

        // Calculate feature contributions under the treatment scenario (T=1)
        let mut x_t1 = Mat::<f32>::zeros(num_rows, num_cols + 1);
        x_t1.as_mut()
            .submatrix_mut(0, 0, num_rows, num_cols)
            .copy_from(x);
        x_t1.as_mut().col_mut(num_cols).fill(1.0);
        let exp_t1 = self.regressor.explain(&x_t1);

        // Calculate feature contributions under the control scenario (T=0)
        let mut x_t0 = Mat::<f32>::zeros(num_rows, num_cols + 1);
        x_t0.as_mut()
            .submatrix_mut(0, 0, num_rows, num_cols)
            .copy_from(x);
        x_t0.as_mut().col_mut(num_cols).fill(0.0);
        let exp_t0 = self.regressor.explain(&x_t0);

        // The difference reveals the source of the treatment effect
        exp_t1 - exp_t0
    }
}
