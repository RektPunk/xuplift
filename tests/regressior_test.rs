use std::sync::Arc;

use faer::{Col, Mat};

pub use xuplift::feature_map::KernelFeatureMap;
pub use xuplift::xmodels::classifier::Classifier;
pub use xuplift::xmodels::regressor::Regressor;

#[test]
fn test_noisy_regression() {
    let n_samples = 500;
    let n_features = 3; // Using 3 features for multi-variable test

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut y = Col::<f32>::zeros(n_samples);

    // 1. Generate Synthetic Multi-variable Data
    // Rule: y = 2.0*x0 - 1.5*x1 + 0.5*x2 + 5.0 (base_value)
    // This generates a predictable non-linear relationship (via cos and power functions)
    // to test the kernel model's ability to approximate the response surface.
    for i in 0..n_samples {
        let v1 = i as f32 * 0.1;
        let v2 = (i as f32 * 0.5).cos();
        let v3 = (i as f32).powi(2) / 1000.0;

        x[(i, 0)] = v1;
        x[(i, 1)] = v2;
        x[(i, 2)] = v3;

        y[i] = 2.0 * v1 - 1.5 * v2 + 0.5 * v3 + 5.0;
    }

    // 2. Setup and Fit Kernel Feature Map
    // The KernelFeatureMap utilizes Nystrom approximation to map input features
    // into a latent space suitable for kernel-based regression.
    let mut map = KernelFeatureMap::new();
    map.fit(&x);
    let map_arc = Arc::new(map);

    // 3. Setup and Fit Regressor
    // Initialize the Regressor with the fitted map and solve for coefficients.
    let mut model = Regressor::new(map_arc);
    model.fit(&y);

    // 4. Verify Prediction Accuracy (MAE)
    // We expect the Mean Absolute Error (MAE) to be low, as the model
    // should accurately recover the underlying generating function.
    let preds = model.predict(&x);
    let mut total_error = 0.0;
    for i in 0..n_samples {
        total_error += (preds[i] - y[i]).abs();
    }
    let mae = total_error / n_samples as f32;
    println!("Multi-variable Regression MAE: {:.4}", mae);
    assert!(mae < 0.5, "Regression MAE is too high: {:.4}", mae);

    // 5. Verify Explanation Consistency
    // Mathematical Consistency Check:
    // The sum of individual feature contributions plus the model's base value (intercept)
    // must exactly equal the final predicted value for every sample.
    let explanation = model.explain(&x);

    // Verify dimensions: rows must match samples, columns must match input features.
    assert_eq!(explanation.nrows(), n_samples, "Rows mismatch");
    assert_eq!(explanation.ncols(), n_features, "Columns mismatch");

    for i in 0..n_samples {
        // Sum of all feature contributions for the current sample
        let mut row_contribution_sum = 0.0;
        for j in 0..n_features {
            row_contribution_sum += explanation[(i, j)];
        }

        // Calculation: Pred(x) == Σ Contribution_j + Intercept
        let reconstructed_pred = row_contribution_sum + model.base_value;

        // Use a small epsilon for floating-point comparison to account for precision loss.
        assert!(
            (reconstructed_pred - preds[i]).abs() < 1e-4,
            "Sample {}: Consistency check failed. Rec: {:.4}, Pred: {:.4}",
            i,
            reconstructed_pred,
            preds[i]
        );
    }
    println!(
        "Explanation consistency check passed for {} features.",
        n_features
    );
}
