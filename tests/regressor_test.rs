use faer::{Col, Mat};

pub use xuplift::feature_map::KernelFeatureMap;
pub use xuplift::xmodels::classifier::Classifier;
pub use xuplift::xmodels::regressor::Regressor;

#[test]
fn test_regression() {
    let n_samples = 500;
    let n_features = 3;
    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut y = Col::<f32>::zeros(n_samples);
    let penalty = 0.01;

    // Generate Synthetic Multi-variable Data
    // Generative Model: y = 2.0*x0 - 1.5*x1 + 0.5*x2 + 5.0 (base_value)
    for i in 0..n_samples {
        let v1 = i as f32 * 0.1;
        let v2 = (i as f32 * 0.5).cos();
        let v3 = (i as f32).powi(2) / 1000.0;

        x[(i, 0)] = v1;
        x[(i, 1)] = v2;
        x[(i, 2)] = v3;

        y[i] = 2.0 * v1 - 1.5 * v2 + 0.5 * v3 + 5.0;
    }

    // Setup and Fit Regressor
    // Initialize the Regressor and solve for coefficients.
    let mut model = Regressor::new(penalty);
    model.fit(x.as_ref(), y.as_ref());

    // Verify Prediction Accuracy (MAE)
    // Expect the Mean Absolute Error (MAE) to be low.
    let preds = model.predict(x.as_ref());
    let mut total_error = 0.0;
    for i in 0..n_samples {
        total_error += (preds[i] - y[i]).abs();
    }
    let mae = total_error / n_samples as f32;
    println!("Multi-variable Regression MAE: {:.4}", mae);
    assert!(mae < 0.1, "Regression MAE is too high: {:.4}", mae);

    // Verify Explanation Consistency
    // The sum of individual feature contributions plus the model's base value (intercept)
    // must exactly equal the predicted value for every sample.
    let explanation = model.explain(x.as_ref());

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

#[test]
fn test_regression_with_nans() {
    let n_samples = 100;
    let n_features = 2;
    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut y = Col::<f32>::zeros(n_samples);

    for i in 0..n_samples {
        x[(i, 0)] = i as f32;
        x[(i, 1)] = if i % 10 == 0 {
            f32::NAN
        } else {
            (i as f32).sin()
        };
        y[i] = x[(i, 0)] + if x[(i, 1)].is_nan() { 0.0 } else { x[(i, 1)] };
    }

    let mut model = Regressor::new(0.01);
    model.fit(x.as_ref(), y.as_ref());

    // Check if predictions for NaN rows are still returning values
    let preds = model.predict(x.as_ref());
    for i in 0..n_samples {
        assert!(
            !preds[i].is_nan(),
            "Prediction should not be NaN for sample {}",
            i
        );
    }
}
