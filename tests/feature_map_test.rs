use faer::Mat;
use xuplift::feature_map::KernelFeatureMap;

#[test]
fn test_feature_map_basic_functionality() {
    // Verify that the KernelFeatureMap initializes and fits correctly on a standard linear dataset.
    let data = Mat::from_fn(10, 2, |r, c| (r as f32) + (c as f32));
    let mut kfm = KernelFeatureMap::new();
    kfm.fit(data.as_ref());

    assert_eq!(kfm.num_features, 2);
    assert!(kfm.num_bases > 0);
    assert_eq!(kfm.feature_bases.len(), 2);
    assert_eq!(kfm.proj_matrices.len(), 2);
    assert_eq!(kfm.feature_means.len(), 2);
    assert_eq!(kfm.s2_invs.len(), 2);
}

#[test]
fn test_transform_batch() {
    // Verify that the batch transform operation produces matrices with the correct dimensions.
    let data = Mat::from_fn(5, 2, |r, c| (r as f32) * (c as f32 + 1.0));
    let mut kfm = KernelFeatureMap::new();
    kfm.fit(data.as_ref());

    let transformed = kfm.transform(data.as_ref());
    assert_eq!(transformed.len(), 2);
    assert_eq!(transformed[0].nrows(), 5);
    assert_eq!(transformed[0].ncols(), kfm.num_bases);
}

#[test]
fn test_transform_row_matches_transform_batch() {
    // Verify that transforming rows individually yields identical results to batch transformation.
    let data = Mat::from_fn(5, 2, |r, c| (r as f32) + (c as f32));
    let mut kfm = KernelFeatureMap::new();
    kfm.fit(data.as_ref());

    let transformed_batch = kfm.transform(data.as_ref());
    for row_idx in 0..5 {
        let transformed_row = kfm.transform_row(data.as_ref(), row_idx);
        for f_idx in 0..2 {
            let offset = f_idx * kfm.num_bases;
            for b in 0..kfm.num_bases {
                let batch_val = transformed_batch[f_idx][(row_idx, b)];
                let row_val = transformed_row[offset + b];
                assert!(
                    (batch_val - row_val).abs() < 1e-5,
                    "Mismatch at row {} feat {} base {}",
                    row_idx,
                    f_idx,
                    b
                );
            }
        }
    }
}

#[test]
fn test_nan_handling_in_feature_map() {
    // Create a matrix with some NaNs to verify robust handling
    let data = Mat::from_fn(4, 1, |r, _| match r {
        0 => 1.0,
        1 => f32::NAN,
        2 => 3.0,
        3 => 4.0,
        _ => 0.0,
    });

    let mut kfm = KernelFeatureMap::new();
    kfm.fit(data.as_ref());

    // Test transform_row with a NaN row
    // Verify that the transformer correctly masks NaNs by outputting a zero vector.
    let x_nan = Mat::from_fn(1, 1, |_, _| f32::NAN);
    let z_nan = kfm.transform_row(x_nan.as_ref(), 0);
    assert!(
        z_nan.iter().all(|&val| val == 0.0),
        "Output for NaN should be zero vector"
    );

    // Test transform_row with a valid row
    let x_valid = Mat::from_fn(1, 1, |_, _| 2.0);
    let z_valid = kfm.transform_row(x_valid.as_ref(), 0);
    assert!(
        z_valid.iter().any(|&val| val != 0.0),
        "Output for valid value should not be zero vector"
    );
}

#[test]
fn test_transform_row_to_slice_correctness() {
    // Explicitly verify that transform_row_to_slice writes correctly to an external buffer.
    use faer::Col;

    let data = Mat::from_fn(3, 2, |r, c| (r + c) as f32);
    let mut kfm = KernelFeatureMap::new();
    kfm.fit(data.as_ref());

    let total_dim = kfm.num_features * kfm.num_bases;
    let mut buffer = Col::<f32>::zeros(total_dim);

    for r in 0..3 {
        // Use the new slice-based transform
        kfm.transform_row_to_slice(data.as_ref(), r, buffer.as_mut());

        // Compare with the standard transform_row
        let expected = kfm.transform_row(data.as_ref(), r);

        for i in 0..total_dim {
            assert_eq!(
                buffer[i], expected[i],
                "Buffer and expected row mismatch at row {} index {}",
                r, i
            );
        }
    }
}
