use faer::{Col, Mat};

use xuplift::metalearners::grlearner::{GRClassifier, GRRegressor};

#[test]
fn test_drclassifier() {
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

        let outcome_noise = if r_idx % 3 == 0 { 0.1 } else { -0.1 };
        let logit = 0.5 * x0 + 0.5 * x1.sin() + (0.8 * treatment) - 0.4 + outcome_noise;
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
    let grclassifier = GRClassifier::new(
        x.as_ref(),
        t.as_ref(),
        y.as_ref(),
        &is_categorical,
        64,
        0.1,
        10,
        0.1,
        0.1,
    );

    let uplift_estimate = grclassifier.predict_uplift(x.as_ref());

    let avg_uplift = uplift_estimate.iter().sum::<f32>() / n_samples as f32;
    let expected_avg_uplift = sum_true_uplift / n_samples as f32;

    assert!(
        (avg_uplift - expected_avg_uplift).abs() < 0.1,
        "Uplift estimation error too high. Est: {:.4}, Expected: {:.4}",
        avg_uplift,
        expected_avg_uplift
    );

    let uplift_explanation = grclassifier.explain_uplift(x.as_ref());
    assert_eq!(uplift_explanation.ncols(), n_features);

    let base_value = grclassifier.tau.base_value;
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

#[test]
fn test_grregressor() {
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

        let treatment_noise = if r_idx % 2 == 0 { 0.5 } else { -0.5 };
        let treatment = 2.0 * x0 + treatment_noise;
        t[r_idx] = treatment;

        let outcome_noise = if r_idx % 3 == 0 { 0.1 } else { -0.1 };
        y[r_idx] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0 + outcome_noise;
    }

    let is_categorical = vec![false; n_features];
    let grregressor = GRRegressor::new(
        x.as_ref(),
        t.as_ref(),
        y.as_ref(),
        &is_categorical,
        64,
        0.001,
        0.001,
        0.001,
    );

    let uplift_estimate = grregressor.predict_uplift(x.as_ref());

    let mut sum_uplift = 0.0;
    for r_idx in 0..n_samples {
        sum_uplift += uplift_estimate[r_idx];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    assert!(
        (avg_uplift - 5.0).abs() < 0.1,
        "Uplift estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    let uplift_explanation = grregressor.explain_uplift(x.as_ref());

    assert_eq!(uplift_explanation.ncols(), n_features);

    let base_value = grregressor.tau.base_value;
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
