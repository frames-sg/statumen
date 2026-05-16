use super::*;

pub struct SlideOpenOptions {
    pub(super) registry: FormatRegistry,
    pub(super) cache_config: CacheConfig,
    pub(super) svcache_policy: crate::formats::svcache::SvcachePolicy,
    pub(super) max_region_pixels: u64,
    pub(super) decode_execution_options: DecodeExecutionOptions,
}

impl SlideOpenOptions {
    pub fn deterministic() -> Self {
        Self {
            registry: FormatRegistry::builtin(),
            cache_config: CacheConfig::deterministic(),
            svcache_policy: crate::formats::svcache::SvcachePolicy::Off,
            max_region_pixels: DEFAULT_MAX_REGION_PIXELS,
            decode_execution_options: DecodeExecutionOptions::default(),
        }
    }

    pub fn with_cache_config(mut self, cache_config: CacheConfig) -> Self {
        self.cache_config = cache_config;
        self
    }

    pub fn with_svcache_policy(
        mut self,
        svcache_policy: crate::formats::svcache::SvcachePolicy,
    ) -> Self {
        self.svcache_policy = svcache_policy;
        self
    }

    pub fn with_registry(mut self, registry: FormatRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn with_max_region_pixels(mut self, max_region_pixels: u64) -> Self {
        self.max_region_pixels = max_region_pixels;
        self
    }

    pub fn with_decode_execution_options(
        mut self,
        decode_execution_options: DecodeExecutionOptions,
    ) -> Self {
        self.decode_execution_options = decode_execution_options;
        self
    }

    pub fn cache_config(&self) -> CacheConfig {
        self.cache_config
    }

    pub fn svcache_policy(&self) -> crate::formats::svcache::SvcachePolicy {
        self.svcache_policy
    }

    pub fn max_region_pixels(&self) -> u64 {
        self.max_region_pixels
    }

    pub fn decode_execution_options(&self) -> DecodeExecutionOptions {
        self.decode_execution_options
    }
}

impl Default for SlideOpenOptions {
    fn default() -> Self {
        Self::deterministic()
    }
}
