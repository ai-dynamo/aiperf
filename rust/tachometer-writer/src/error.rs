use thiserror::Error;

#[derive(Error, Debug)]
pub enum NoMoreError {
    #[error("S3 error: {0}")]
    S3(String),

    #[error("Arrow error: {0}")]
    Arrow(String),

    #[error("Parquet error: {0}")]
    Parquet(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid schema: {0}")]
    InvalidSchema(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),
}

pub type Result<T> = std::result::Result<T, NoMoreError>;
