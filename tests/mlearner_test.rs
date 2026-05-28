use faer::{Col, Mat};

use xuplift::metalearners::mlearner::MLearner;

#[test]
fn test_mlearner() {
    let n_samples = 500;
    let n_features = 3;

    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut t = Col::<f32>::zeros(n_samples);
    let mut y = Col::<f32>::zeros(n_samples);

    // Synthetic Data Generation
    // Objective: Create a dataset with a known constant treatment effect.
    // Generative Model: y = 1.5*x0 + 0.5*sin(x1) + (5.0 * t) + 10.0
    // Ground Truth Uplift (ITE): 5.0
    for i in 0..n_samples {
        let x0 = i as f32 * 0.01;
        let x1 = (i as f32).sin();
        let x2 = (i as f32).cos();

        x[(i, 0)] = x0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;

        // Assign treatment: Even indices = Treatment (1), Odd indices = Control (0)
        // M-Learner assumes a randomized controlled trial (RCT) environment with 50/50 propensity.
        let treatment = if i % 2 == 0 { 1.0 } else { 0.0 };
        t[i] = treatment;
        y[i] = 1.5 * x0 + 0.5 * x1.sin() + (5.0 * treatment) + 10.0;
    }

    // Initialize MLearner which internally applies target transformation
    // and trains a single tau model directly on the modified target.
    let mlearner = MLearner::new(x.as_ref(), t.as_ref(), y.as_ref(), 0.1);

    // Estimate Individual Treatment Effect (ITE).
    let uplift_estimate = mlearner.predict_uplift(x.as_ref());

    // Verify if the average estimated uplift is close to the true effect.
    let mut sum_uplift = 0.0;
    for i in 0..n_samples {
        sum_uplift += uplift_estimate[i];
    }
    let avg_uplift = sum_uplift / n_samples as f32;

    println!(
        "True Uplift: 5.0, M-Learner Estimated Average Uplift: {:.4}",
        avg_uplift
    );

    assert!(
        (avg_uplift - 5.0).abs() < 0.5,
        "Uplift estimation is too far from ground truth. Got: {:.4}",
        avg_uplift
    );

    // Verify Mathematical Explanation Consistency
    // In M-Learner, the explanation is straightforward because it uses a single tau regressor.
    // The sum of features plus the static base_value must equal the prediction.
    let uplift_explanation = mlearner.explain_uplift(x.as_ref());

    // M-Learner's explanation matrix should have n_features columns.
    assert_eq!(uplift_explanation.ncols(), n_features);

    // Extract the static base value once outside the loop for mathematical correctness
    let base_value = mlearner.tau.base_value;

    for i in 0..x.nrows() {
        let mut explained_total = 0.0;
        for j in 0..uplift_explanation.ncols() {
            explained_total += uplift_explanation[(i, j)];
        }

        // The sum of contributions + base must equal the explained uplift
        let total_reconstructed_uplift = explained_total + base_value;
        assert!(
            (total_reconstructed_uplift - uplift_estimate[i]).abs() < 1e-4,
            "Uplift explanation delta mismatch at sample {}: Explained {:.4}, Predicted {:.4}",
            i,
            total_reconstructed_uplift,
            uplift_estimate[i]
        );
    }
    println!("MLearner Uplift Delta Explanation check passed!");
}
