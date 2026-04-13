use faer::{Col, Mat};

// Helper to create sliced matrices
pub fn filter_rows(x: &Mat<f32>, indices: &[usize]) -> Mat<f32> {
    let mut filtered = Mat::<f32>::zeros(indices.len(), x.ncols());
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        filtered
            .as_mut()
            .row_mut(new_idx)
            .copy_from(x.as_ref().row(old_idx));
    }
    filtered
}

// Helper to create sliced column vectors
pub fn filter_cols_vec(y: &Col<f32>, indices: &[usize]) -> Col<f32> {
    let mut filtered = Col::<f32>::zeros(indices.len());
    for (new_idx, &old_idx) in indices.iter().enumerate() {
        filtered[new_idx] = y[old_idx];
    }
    filtered
}
