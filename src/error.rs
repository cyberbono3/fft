use thiserror::Error;

#[derive(Debug, Error)]
pub enum FftError {
    #[error("Input length ({0}) is not a power of two.")]
    NotAPowerOfTwo(usize),
}
