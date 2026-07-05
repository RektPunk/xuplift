use faer::{Col, Mat};

use xuplift::metalearners::rlearner::RRegressor;

#[test]
fn test_rregressor() {
    let n_samples = 500;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    // --- Synthetic Data Generation ---
    // Objective: Create a dataset with a known constant treatment effect.
    // Generative Model: y = 1.5*x0 + 0.5*sin(x1) + (5.0 * t) + 10.0
    // Ground Truth Uplift (ITE): 5.0
    for r_idx in 0..n_samples {
        let x0 = r_idx as f32 * 0.01;
        let x1 = (r_idx as f32).sin();
        let x2 = (r_idx as f32).cos();

        x[(r_idx, 0)] = x0;
        x[(r_idx, 1)] = x1;
        x[(r_idx, 2)] = x2;

        // Random treatment assignment
        let treatment = if r_idx % 2 == 0 { 1.0 } else { 0.0 };
        t[r_idx] = treatment;

        // Outcome = Baseline + (Treatment * Effect) + Noise
        y[r_idx] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0;
    }

    // --- Model Initialization ---
    // RRegressor fits: m(x) [Outcome], e(x) [Propensity], and tau(x) [Residual-on-Residual]
    let is_categorical = vec![false; n_features];
    let rregressor = RRegressor::new(
        x.as_ref(),
        t.as_ref(),
        y.as_ref(),
        &is_categorical,
        64,
        0.1,
        0.1,
        20,
        0.1,
    );

    // --- Prediction ---
    let uplift_estimate = rregressor.predict_uplift(x.as_ref());

    // --- Verification: Accuracy ---
    let mut sum_uplift = 0.0;
    for r_idx in 0..n_samples {
        sum_uplift += uplift_estimate[r_idx];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, RRegressor Estimated Avg Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 5.0).abs() < 0.1,
        "RRegressor estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    // --- Verification: Explanation Consistency ---
    let uplift_explanation = rregressor.explain_uplift(x.as_ref());

    // R-Learner's explanation matrix should have n_features columns.
    assert_eq!(uplift_explanation.ncols(), n_features);

    for r_idx in 0..x.nrows() {
        let mut explained_total = 0.0;
        for p_idx in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(r_idx, p_idx)];
        }

        // Reconstructed Uplift = sum(feature_contributions) + base_value
        let total_reconstructed_uplift = explained_total + rregressor.tau.base_value;
        assert!(
            (total_reconstructed_uplift - uplift_estimate[r_idx]).abs() < 1e-4,
            "RRegressor explanation mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            r_idx,
            total_reconstructed_uplift,
            uplift_estimate[r_idx]
        );
    }
    println!("RRegressor verification passed!");
}
