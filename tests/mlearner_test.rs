use faer::{Col, Mat};

use xuplift::metalearners::mlearner::MRegressor;

#[test]
fn test_mregressor() {
    let n_samples = 500;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    for r_idx in 0..n_samples {
        let x0 = r_idx as f32 * 0.01;
        let x1 = (r_idx as f32).sin();
        let x2 = (r_idx as f32).cos();

        x[(r_idx, 0)] = x0;
        x[(r_idx, 1)] = x1;
        x[(r_idx, 2)] = x2;

        let treatment = if r_idx % 2 == 0 { 1.0 } else { 0.0 };
        t[r_idx] = treatment;
        y[r_idx] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0;
    }

    let is_categorical = vec![false; n_features];
    let mregressor = MRegressor::new(x.as_ref(), t.as_ref(), y.as_ref(), &is_categorical, 64, 0.1);

    let uplift_estimate = mregressor.predict_uplift(x.as_ref());

    let mut sum_uplift = 0.0;
    for r_idx in 0..n_samples {
        sum_uplift += uplift_estimate[r_idx];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    assert!(
        (avg_uplift - 5.0).abs() < 0.5,
        "Uplift estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    let uplift_explanation = mregressor.explain_uplift(x.as_ref());
    assert_eq!(uplift_explanation.ncols(), n_features);

    let base_value = mregressor.tau.base_value;
    for r_idx in 0..x.nrows() {
        let mut explained_total = 0.0;
        for p_idx in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(r_idx, p_idx)];
        }

        let total_reconstructed_uplift = explained_total + base_value;
        assert!(
            (total_reconstructed_uplift - uplift_estimate[r_idx]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            r_idx,
            total_reconstructed_uplift,
            uplift_estimate[r_idx]
        );
    }
}
