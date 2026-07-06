use faer::{Col, Mat};

use xuplift::metalearners::xlearner::{XClassifier, XRegressor};

#[test]
fn test_xclassifier() {
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

        let treatment = if r_idx % 5 == 0 { 1.0 } else { 0.0 };
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
    let xclassifier = XClassifier::new(
        x.as_ref(),
        t.as_ref(),
        y.as_ref(),
        &is_categorical,
        64,
        0.1,
        10,
        0.1,
        20,
        0.1,
    );

    let uplift_estimate = xclassifier.predict_uplift(x.as_ref());

    let avg_uplift = uplift_estimate.iter().sum::<f32>() / n_samples as f32;
    let expected_avg_uplift = sum_true_uplift / n_samples as f32;

    assert!(
        (avg_uplift - expected_avg_uplift).abs() < 0.1,
        "Uplift estimation error too high. Est: {:.4}, Expected: {:.4}",
        avg_uplift,
        expected_avg_uplift
    );

    let uplift_explanation = xclassifier.explain_uplift(x.as_ref());
    assert_eq!(uplift_explanation.ncols(), n_features);

    let p_pred = xclassifier.p.predict(x.as_ref());

    for r_idx in 0..x.nrows() {
        let mut explained_total = 0.0;
        for p_idx in 0..n_features {
            explained_total += uplift_explanation[(r_idx, p_idx)];
        }

        let gi = p_pred[r_idx].clamp(0.01, 0.99);
        let dynamic_base =
            gi * xclassifier.tau_t0.base_value + (1.0 - gi) * xclassifier.tau_t1.base_value;

        let total_reconstructed_uplift = explained_total + dynamic_base;
        assert!(
            (total_reconstructed_uplift - uplift_estimate[r_idx]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            r_idx,
            total_reconstructed_uplift,
            uplift_estimate[r_idx]
        );
    }
}

#[test]
fn test_xregressor() {
    let n_samples = 500;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    for r_idx in 0..n_samples {
        let x0 = r_idx as f32 * 0.02;
        let x1 = (r_idx as f32 * 0.1).cos();
        let x2 = (r_idx as f32 * 0.1).sin();

        x[(r_idx, 0)] = x0;
        x[(r_idx, 1)] = x1;
        x[(r_idx, 2)] = x2;

        let treatment = if r_idx % 5 == 0 { 1.0 } else { 0.0 };
        t[r_idx] = treatment;

        y[r_idx] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0;
    }

    let is_categorical = vec![false; n_features];
    let xregressor = XRegressor::new(
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

    let uplift_estimate = xregressor.predict_uplift(x.as_ref());

    let mut sum_uplift = 0.0;
    for r_idx in 0..n_samples {
        sum_uplift += uplift_estimate[r_idx];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    assert!(
        (avg_uplift - 5.0).abs() < 0.1,
        "Uplift estimation error too high. Got: {:.4}",
        avg_uplift
    );

    let uplift_explanation = xregressor.explain_uplift(x.as_ref());
    assert_eq!(uplift_explanation.ncols(), n_features);

    let p_pred = xregressor.p.predict(x.as_ref());

    for r_idx in 0..x.nrows() {
        let mut explained_total = 0.0;
        for p_idx in 0..n_features {
            explained_total += uplift_explanation[(r_idx, p_idx)];
        }

        let gi = p_pred[r_idx].clamp(0.01, 0.99);
        let dynamic_base =
            gi * xregressor.tau_t0.base_value + (1.0 - gi) * xregressor.tau_t1.base_value;

        let total_reconstructed_uplift = explained_total + dynamic_base;
        assert!(
            (total_reconstructed_uplift - uplift_estimate[r_idx]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            r_idx,
            total_reconstructed_uplift,
            uplift_estimate[r_idx]
        );
    }
}
