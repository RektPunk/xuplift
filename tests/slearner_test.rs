use faer::{Col, Mat};

use xuplift::metalearners::slearner::{SClassifier, SRegressor};

#[test]
fn test_sclassifier() {
    let n_samples = 500;
    let n_features = 3;
    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);
    let mut sum_true_uplift = 0.0;

    for r_idx in 0..n_samples {
        let x0 = r_idx as f32 * 0.02;
        let x1 = (r_idx as f32 * 0.1).cos();
        let x2 = (r_idx as f32 * 0.1).sin();
        x[(r_idx, 0)] = x0;
        x[(r_idx, 1)] = x1;
        x[(r_idx, 2)] = x2;

        let treatment = if r_idx % 2 == 0 { 1.0 } else { 0.0 };
        t[r_idx] = treatment;

        let logit = 0.5 * x0 + 0.5 * x1.sin() + (0.8 * treatment) - 0.4;
        let prob = 1.0 / (1.0 + (-logit).exp());
        y[r_idx] = if rand::random_range(0.0..1.0) < prob {
            1.0
        } else {
            0.0
        };

        let logit_t1 = 0.5 * x0 + 0.5 * x1.sin() + 0.8 - 0.4;
        let logit_t0 = 0.5 * x0 + 0.5 * x1.sin() - 0.4;

        let prob_t1 = 1.0 / (1.0 + (-logit_t1).exp());
        let prob_t0 = 1.0 / (1.0 + (-logit_t0).exp());

        let true_uplift = prob_t1 - prob_t0;
        sum_true_uplift += true_uplift;
    }

    let is_categorical = vec![false; n_features];
    let sclassifier = SClassifier::new(
        x.as_ref(),
        t.as_ref(),
        y.as_ref(),
        &is_categorical,
        64,
        0.1,
        10,
    );

    let uplift_estimate = sclassifier.predict_uplift(x.as_ref());
    let avg_uplift = uplift_estimate.iter().sum::<f32>() / n_samples as f32;
    let expected_avg_uplift = sum_true_uplift / n_samples as f32;
    assert!(
        (avg_uplift - expected_avg_uplift).abs() < 0.1,
        "Uplift estimation error too high. Est: {:.4}, Expected: {:.4}",
        avg_uplift,
        expected_avg_uplift
    );

    let uplift_explanation = sclassifier.explain_uplift(x.as_ref());
    assert_eq!(uplift_explanation.ncols(), n_features + 1);

    let mut scratch = Mat::<f32>::zeros(n_samples, n_features + 1);
    scratch
        .as_mut()
        .submatrix_mut(0, 0, n_samples, n_features)
        .copy_from(x.as_ref());

    scratch.as_mut().col_mut(n_features).fill(1.0);
    let p1_col = sclassifier.mu.predict(scratch.as_ref());

    scratch.as_mut().col_mut(n_features).fill(0.0);
    let p0_col = sclassifier.mu.predict(scratch.as_ref());

    for r_idx in 0..n_features {
        let mut explained_total = 0.0;
        for p_idx in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(r_idx, p_idx)];
        }
        let p1 = p1_col[r_idx];
        let p0 = p0_col[r_idx];

        let logit_t1 = (p1 / (1.0 - p1)).ln();
        let logit_t0 = (p0 / (1.0 - p0)).ln();
        let model_logit_delta = logit_t1 - logit_t0;
        assert!(
            (explained_total - model_logit_delta).abs() < 1e-4,
            "Row {}: SHAP sum ({}) != Model logit delta ({})",
            r_idx,
            explained_total,
            model_logit_delta
        );

        assert!(
            ((p1 - p0) - uplift_estimate[r_idx]).abs() < 1e-6,
            "Row {}: Prediction discrepancy",
            r_idx
        );
    }
}

#[test]
fn test_sregressor() {
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

    // --- Model Initialization ---
    let is_categorical = vec![false; n_features + 1];
    let sregressor = SRegressor::new(
        x.as_ref(),
        t.as_ref(),
        y.as_ref(),
        &is_categorical,
        64,
        0.01,
    );

    // --- Prediction ---
    let uplift_estimate = sregressor.predict_uplift(x.as_ref());

    // --- Verification: Accuracy ---
    let mut sum_uplift = 0.0;
    for r_idx in 0..n_samples {
        sum_uplift += uplift_estimate[r_idx];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, Estimated Average Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 5.0).abs() < 0.1,
        "Uplift estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    let uplift_explanation = sregressor.explain_uplift(x.as_ref());
    assert_eq!(uplift_explanation.ncols(), n_features + 1);

    for r_idx in 0..x.nrows() {
        let mut explained_total = 0.0;
        for p_idx in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(r_idx, p_idx)];
        }
        assert!(
            (explained_total - uplift_estimate[r_idx]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            r_idx,
            explained_total,
            uplift_estimate[r_idx]
        );
    }
}
