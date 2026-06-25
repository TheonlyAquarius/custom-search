cd ~/ck
git checkout -b qwen3-embed

# 1) New provider file (whole file, no line numbers to worry about)
cat > ck-embed/src/qwen3.rs <<'RUST'
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
RUST

# 2) The three in-place edits, done as exact-string swaps with safety checks.
#    If any anchor doesn't match, it aborts loudly instead of corrupting a file.
python3 - <<'PY'
import sys

def patch(path, old, new):
    s = open(path).read()
    n = s.count(old)
    if n != 1:
        sys.exit(f"ABORT: expected exactly 1 match in {path}, found {n}. No changes written.")
    open(path, "w").write(s.replace(old, new))
    print(f"patched {path}")

# --- ck-embed/src/lib.rs : register the module ---
patch("ck-embed/src/lib.rs",
'''#[cfg(feature = "mixedbread")]
mod mixedbread;
#[cfg(feature = "mixedbread")]
use mixedbread::MixedbreadEmbedder;''',
'''#[cfg(feature = "mixedbread")]
mod mixedbread;
#[cfg(feature = "mixedbread")]
use mixedbread::MixedbreadEmbedder;

#[cfg(feature = "qwen3")]
mod qwen3;
#[cfg(feature = "qwen3")]
use qwen3::Qwen3Embedder;''')

# --- ck-embed/src/lib.rs : add the dispatch arm ---
patch("ck-embed/src/lib.rs",
'''        provider => bail!("Unsupported embedding provider '{provider}'"),''',
'''        "qwen3" => {
            #[cfg(feature = "qwen3")]
            {
                return Ok(Box::new(Qwen3Embedder::new(config, progress_callback)?));
            }
            #[cfg(not(feature = "qwen3"))]
            {
                bail!(
                    "Model '{}' requires the `qwen3` feature. Rebuild ck with Qwen3 support.",
                    config.name
                );
            }
        }
        provider => bail!("Unsupported embedding provider '{provider}'"),''')

# --- ck-embed/Cargo.toml : add candle-core dep ---
patch("ck-embed/Cargo.toml",
'''num_cpus = { workspace = true, optional = true }''',
'''num_cpus = { workspace = true, optional = true }
candle-core = { version = "0.10.2", optional = true }''')

# --- ck-embed/Cargo.toml : add the qwen3 feature + enable by default ---
patch("ck-embed/Cargo.toml",
'''default = ["fastembed", "mixedbread"]
fastembed = ["dep:fastembed"]''',
'''default = ["fastembed", "mixedbread", "qwen3"]
fastembed = ["dep:fastembed"]
qwen3 = ["dep:fastembed", "dep:candle-core", "fastembed/qwen3"]''')

# --- ck-models/src/lib.rs : register the qwen3-0.6b alias ---
patch("ck-models/src/lib.rs",
'''                description: "Mixedbread xsmall embedding model (4k context, 384 dims) optimized for local semantic search".to_string(),
            },
        );''',
'''                description: "Mixedbread xsmall embedding model (4k context, 384 dims) optimized for local semantic search".to_string(),
            },
        );

        models.insert(
            "qwen3-0.6b".to_string(),
            ModelConfig {
                name: "Qwen/Qwen3-Embedding-0.6B".to_string(),
                provider: "qwen3".to_string(),
                dimensions: 1024,
                max_tokens: 8192,
                description: "Qwen3 0.6B text embedding (candle backend, 1024 dims)".to_string(),
            },
        );''')

print("all edits applied.")
PY

# 3) Build it (candle compiles for the first time here — expect several minutes)
cargo build --release -p ck-search
