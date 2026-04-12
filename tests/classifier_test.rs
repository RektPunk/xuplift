use faer::{Col, Mat};
use rand::RngExt;

pub use xuplift::feature_map::KernelFeatureMap;
pub use xuplift::xmodels::classifier::Classifier;
pub use xuplift::xmodels::regressor::Regressor;
#[test]
fn test_gaussian_classification() {
    let mut rng = rand::rng();
    let n_samples = 300;
    let n_features = 2;
    let mut x = Mat::<f32>::zeros(n_samples, n_features);
    let mut y = Col::<f32>::zeros(n_samples);

    // Generates a sample from a normal distribution with given mean and std_dev.
    let mut sample_normal = |mean: f32, std_dev: f32| -> f32 {
        let u1: f32 = rng.random_range(0.0..1.0); // Uniform(0, 1)
        let u2: f32 = rng.random_range(0.0..1.0);
        // Standard normal (mean=0, std=1)
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        z0 * std_dev + mean
    };

    // Generate Gaussian Blobs
    for i in 0..n_samples {
        if i < n_samples / 2 {
            x[(i, 0)] = sample_normal(-1.5, 0.7);
            x[(i, 1)] = sample_normal(-1.5, 0.7);
            y[i] = 0.0;
        } else {
            x[(i, 0)] = sample_normal(1.5, 0.7);
            x[(i, 1)] = sample_normal(1.5, 0.7);
            y[i] = 1.0;
        }
    }

    // 2. Setup and Fit Kernel Feature Map
    // Project input features into a high-dimensional space using the Nystrom approximation.
    let mut map = KernelFeatureMap::new();
    map.fit(&x);

    // 3. Setup and Fit Classifier (IRLS)
    // Train a Logistic Regression model using Iteratively Reweighted Least Squares (IRLS).
    let mut model = Classifier::new(map);
    model.fit(&y, 20); // Perform 20 iterations for convergence

    // 4. Verify Accuracy
    // Ensure that the model can linearly separate the kernel-mapped Gaussian blobs.
    let probs = model.predict(&x);
    let mut correct = 0;
    for i in 0..n_samples {
        let pred = if probs[i] > 0.5 { 1.0 } else { 0.0 };
        if (pred - y[i]).abs() < 1e-5 {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / n_samples as f32;
    assert!(
        accuracy > 0.95,
        "Accuracy is too low: {:.2}%",
        accuracy * 100.0
    );

    // 5. Verify Explanation Consistency
    // In Logistic Regression, the explanation provides feature contributions in the logit space.
    // We verify that: Sigmoid(Sum(Contributions) + Base_Value) == Predicted_Probability
    let explanation = model.explain(&x);

    for i in 0..n_samples {
        // Sum of logit-scale contributions for each feature
        let mut logit_sum = 0.0;
        for j in 0..n_features {
            logit_sum += explanation[(i, j)];
        }

        // Add the global bias (intercept) to the sum of contributions
        let total_logit = logit_sum + model.base_value;

        // Reconstruct probability using the Sigmoid function: 1 / (1 + exp(-logit))
        let reconstructed_prob = 1.0 / (1.0 + (-total_logit).exp());

        // Check if the reconstructed probability matches the model's direct prediction
        assert!(
            (reconstructed_prob - probs[i]).abs() < 1e-4,
            "Sample {}: Explanation consistency failed. Rec: {:.4}, Prob: {:.4}",
            i,
            reconstructed_prob,
            probs[i]
        );
    }
    println!("Classification explanation check passed for Gaussian Blobs.");
}
