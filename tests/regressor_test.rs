use faer::{Col, Mat};

pub use xuplift::xmodels::classifier::Classifier;
pub use xuplift::xmodels::feature_map::KernelFeatureMap;
pub use xuplift::xmodels::regressor::Regressor;

#[test]
fn test_regression() {
    let n_samples = 500;
    let n_features = 3;
    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut y = Col::<f32>::zeros(n_samples);
    let penalty = 0.01;

    for r_idx in 0..n_samples {
        let v1 = r_idx as f32 * 0.1;
        let v2 = (r_idx as f32 * 0.5).cos();
        let v3 = (r_idx as f32).powi(2) / 1000.0;

        x[(r_idx, 0)] = v1;
        x[(r_idx, 1)] = v2;
        x[(r_idx, 2)] = v3;

        y[r_idx] = 2.0 * v1 - 1.5 * v2 + 0.5 * v3 + 5.0;
    }

    let mut model = Regressor::new(64, penalty);
    let is_categorical = vec![false; n_features];
    model.fit(x.as_ref(), y.as_ref(), &is_categorical);

    let y_pred = model.predict(x.as_ref());
    let mut total_error = 0.0;
    for r_idx in 0..n_samples {
        total_error += (y_pred[r_idx] - y[r_idx]).abs();
    }
    let mae = total_error / n_samples as f32;
    println!("Multi-variable Regression MAE: {:.4}", mae);
    assert!(mae < 0.1, "Regression MAE is too high: {:.4}", mae);

    let uplift_explanation = model.explain(x.as_ref());

    assert_eq!(uplift_explanation.nrows(), n_samples, "Rows mismatch");
    assert_eq!(uplift_explanation.ncols(), n_features, "Columns mismatch");

    for r_idx in 0..n_samples {
        let mut row_contribution_sum = 0.0;
        for p_idx in 0..n_features {
            row_contribution_sum += uplift_explanation[(r_idx, p_idx)];
        }

        let total_reconstructed_pred = row_contribution_sum + model.base_value;
        assert!(
            (total_reconstructed_pred - y_pred[r_idx]).abs() < 1e-4,
            "Sample {}: Consistency check failed. Rec: {:.4}, Pred: {:.4}",
            r_idx,
            total_reconstructed_pred,
            y_pred[r_idx]
        );
    }
}

#[test]
fn test_regression_with_nans() {
    let n_samples = 100;
    let n_features = 2;
    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut y = Col::<f32>::zeros(n_samples);

    for r_idx in 0..n_samples {
        x[(r_idx, 0)] = r_idx as f32;
        x[(r_idx, 1)] = if r_idx % 10 == 0 {
            f32::NAN
        } else {
            (r_idx as f32).sin()
        };
        y[r_idx] = x[(r_idx, 0)]
            + if x[(r_idx, 1)].is_nan() {
                0.0
            } else {
                x[(r_idx, 1)]
            };
    }

    let mut model = Regressor::new(64, 0.01);
    let is_categorical = vec![false; n_features];
    model.fit(x.as_ref(), y.as_ref(), &is_categorical);

    let y_pred = model.predict(x.as_ref());
    for r_idx in 0..n_samples {
        assert!(
            !y_pred[r_idx].is_nan(),
            "Prediction should not be NaN for sample {}",
            r_idx
        );
    }
}
