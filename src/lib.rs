pub mod feature_map;
pub mod xmodels;

pub use crate::feature_map::KernelFeatureMap;
pub use crate::xmodels::classifier::Classifier;
pub use crate::xmodels::regressor::Regressor;

#[cfg(test)]
mod tests {
    use super::*;
    use faer::{Col, Mat};
    #[test]
    fn test_gaussian_classification() {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rng();
        let n_samples = 300;
        let n_features = 2;
        let mut x = Mat::<f32>::zeros(n_samples, n_features);
        let mut y = Col::<f32>::zeros(n_samples);

        // 1. Generate Gaussian Blobs
        let dist1 = Normal::new(-1.5, 0.7).unwrap();
        let dist2 = Normal::new(1.5, 0.7).unwrap();

        for i in 0..n_samples {
            if i < n_samples / 2 {
                x[(i, 0)] = dist1.sample(&mut rng);
                x[(i, 1)] = dist1.sample(&mut rng);
                y[i] = 0.0;
            } else {
                x[(i, 0)] = dist2.sample(&mut rng);
                x[(i, 1)] = dist2.sample(&mut rng);
                y[i] = 1.0;
            }
        }

        // 2. Setup and Fit Kernel Feature Map
        let mut map = KernelFeatureMap::new();
        map.fit(&x);

        // 3. Setup and Fit Classifier (IRLS)
        let mut model = Classifier::new(map);
        model.fit(&y, 20);

        // 4. Verify Accuracy
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
        let explanation = model.explain(&x);

        for i in 0..n_samples {
            // Sum of logit contributions for each feature
            let mut logit_sum = 0.0;
            for j in 0..n_features {
                logit_sum += explanation[(i, j)];
            }

            // In Logistic Regression, the sum of contributions + base_value
            // should equal the logit (inverse sigmoid of the probability).
            let total_logit = logit_sum + model.base_value;
            let reconstructed_prob = 1.0 / (1.0 + (-total_logit).exp()); // Sigmoid function

            // Check if the reconstructed probability matches the predicted probability
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

    #[test]
    fn test_noisy_regression() {
        use super::*;
        let n_samples = 100;
        let n_features = 3; // Using 3 features for multi-variable test

        let mut x = Mat::<f32>::zeros(n_samples, n_features);
        let mut y = Col::<f32>::zeros(n_samples);

        // 1. Generate Synthetic Multi-variable Data
        // Rule: y = 2.0*x0 - 1.5*x1 + 0.5*x2 + 5.0 (base_value)
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
        let mut map = KernelFeatureMap::new();
        map.fit(&x);

        // 3. Setup and Fit Regressor
        let mut model = Regressor::new(map);
        model.fit(&y);

        // 4. Verify Prediction Accuracy (MAE)
        let preds = model.predict(&x);
        let mut total_error = 0.0;
        for i in 0..n_samples {
            total_error += (preds[i] - y[i]).abs();
        }
        let mae = total_error / n_samples as f32;
        println!("Multi-variable Regression MAE: {:.4}", mae);
        assert!(mae < 0.5, "Regression MAE is too high: {:.4}", mae);

        // 5. Verify Explanation Consistency
        let explanation = model.explain(&x);

        // Check dimensions: rows should be samples, columns should be features
        assert_eq!(explanation.nrows(), n_samples, "Rows mismatch");
        assert_eq!(explanation.ncols(), n_features, "Columns mismatch");

        for i in 0..n_samples {
            // Sum of all feature contributions for the current sample
            let mut row_contribution_sum = 0.0;
            for j in 0..n_features {
                row_contribution_sum += explanation[(i, j)];
            }

            // Mathematical consistency: Sum(contributions) + base_value == prediction
            let reconstructed_pred = row_contribution_sum + model.base_value;

            // Using small epsilon for floating point comparison
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
}
