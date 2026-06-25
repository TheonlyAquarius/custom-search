use anyhow::{Context, Result};
use candle_core::{DType, Device};
use fastembed::Qwen3TextEmbedding;

use crate::{Embedder, ModelDownloadCallback};
use ck_models::ModelConfig;

/// Candle-backed Qwen3 text-embedding provider.
///
/// Wraps `fastembed::Qwen3TextEmbedding`, which downloads the Hugging Face
/// checkpoint named in `config.name` (e.g. `Qwen/Qwen3-Embedding-0.6B`),
/// then handles tokenization, pooling, and L2-normalization internally.
pub struct Qwen3Embedder {
    inner: Qwen3TextEmbedding,
    dim: usize,
    model_name: String,
}

impl Qwen3Embedder {
    pub fn new(
        config: &ModelConfig,
        progress_callback: Option<ModelDownloadCallback>,
    ) -> Result<Self> {
        if let Some(cb) = progress_callback.as_ref() {
            cb(&format!(
                "Loading Qwen3 embedding model ({}) via candle (downloads on first use)...",
                config.name
            ));
        }

        // Uses the GPU if this binary was built with `fastembed/cuda`; otherwise
        // falls back to CPU transparently.
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        let inner = Qwen3TextEmbedding::from_hf(
            config.name.as_str(),
            &device,
            DType::F32,
            config.max_tokens,
        )
        .with_context(|| format!("failed to load Qwen3 model '{}'", config.name))?;

        if let Some(cb) = progress_callback.as_ref() {
            cb("Qwen3 model loaded successfully");
        }

        Ok(Self {
            inner,
            dim: config.dimensions,
            model_name: config.name.clone(),
        })
    }
}

impl Embedder for Qwen3Embedder {
    fn id(&self) -> &'static str {
        "qwen3"
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn embed(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        self.inner
            .embed(texts)
            .map_err(|e| anyhow::anyhow!("Qwen3 embedding failed: {e}"))
    }
}
