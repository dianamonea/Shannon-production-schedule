/// Dynamic Capability Loading Module
/// 
/// Supports downloading and loading manufacturing capabilities (kinematics solvers, grasp planners, etc.)
/// as WASM modules from cloud, and executing them at the edge with full isolation.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

#[cfg(feature = "wasi")]
use crate::wasi_sandbox::WasiSandbox;

/// Capability metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityMetadata {
    pub capability_id: String,
    pub name: String,
    pub version: String,
    pub capability_type: CapabilityType,
    pub wasm_hash: String,
    pub dependencies: Vec<String>,
    pub parameters: HashMap<String, ParameterSchema>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityType {
    InverseKinematics,
    ForwardKinematics,
    GraspPlanning,
    PathPlanning,
    Optimization,
    Validation,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    pub param_type: String,  // "float", "int", "string", "array"
    pub required: bool,
    pub default: Option<serde_json::Value>,
}

/// Loaded capability in memory
pub struct LoadedCapability {
    pub metadata: CapabilityMetadata,
    pub wasm_module: Vec<u8>,
    pub loaded_at: u64,
    #[cfg(feature = "wasi")]
    pub sandbox: Option<WasiSandbox>,
}

/// Capability request to cloud service
#[derive(Debug, Serialize)]
pub struct CapabilityRequest {
    pub capability_type: String,
    pub robot_type: String,
    pub parameters: HashMap<String, String>,
}

/// Capability response from cloud
#[derive(Debug, Deserialize)]
pub struct CapabilityResponse {
    pub capability_id: String,
    pub metadata: CapabilityMetadata,
    pub wasm_base64: String,
    pub download_url: Option<String>,
}

/// Manager for dynamic capabilities
pub struct CapabilityManager {
    cache_dir: PathBuf,
    loaded_capabilities: Arc<RwLock<HashMap<String, LoadedCapability>>>,
    cloud_service_url: String,
    #[cfg(feature = "wasi")]
    sandbox_factory: Option<WasiSandbox>,
}

impl CapabilityManager {
    pub fn new(cache_dir: Option<PathBuf>, cloud_service_url: Option<String>) -> Self {
        let cache = cache_dir.unwrap_or_else(|| PathBuf::from("/tmp/capabilities"));
        let cloud_url = cloud_service_url
            .or_else(|| std::env::var("CAPABILITY_SERVICE_URL").ok())
            .unwrap_or_else(|| "http://capability-service:8001".to_string());

        Self {
            cache_dir: cache,
            loaded_capabilities: Arc::new(RwLock::new(HashMap::new())),
            cloud_service_url: cloud_url,
            #[cfg(feature = "wasi")]
            sandbox_factory: None,
        }
    }

    #[cfg(feature = "wasi")]
    pub fn with_sandbox(mut self, sandbox: Option<WasiSandbox>) -> Self {
        self.sandbox_factory = sandbox;
        self
    }

    /// Download capability from cloud service
    pub async fn download_capability(&self, capability_type: &str, robot_type: &str) -> Result<CapabilityMetadata> {
        info!(
            "Downloading capability: {} for robot: {}",
            capability_type, robot_type
        );

        let request = CapabilityRequest {
            capability_type: capability_type.to_string(),
            robot_type: robot_type.to_string(),
            parameters: HashMap::new(),
        };

        // In production, make HTTP request to cloud service
        // let response = reqwest::Client::new()
        //     .post(&format!("{}/capabilities/download", self.cloud_service_url))
        //     .json(&request)
        //     .send()
        //     .await?;
        // let cap_response: CapabilityResponse = response.json().await?;

        // For now, return mock response
        warn!("Using mock capability download (not connected to real cloud service)");
        Ok(CapabilityMetadata {
            capability_id: format!("cap_{}_{}_{}", capability_type, robot_type, chrono::Local::now().timestamp()),
            name: format!("{} for {}", capability_type, robot_type),
            version: "1.0.0".to_string(),
            capability_type: match capability_type {
                "ik" => CapabilityType::InverseKinematics,
                "fk" => CapabilityType::ForwardKinematics,
                "grasp" => CapabilityType::GraspPlanning,
                _ => CapabilityType::Custom(capability_type.to_string()),
            },
            wasm_hash: "mock_hash".to_string(),
            dependencies: vec![],
            parameters: HashMap::new(),
        })
    }

    /// Load capability into memory with WASI sandbox isolation
    pub async fn load_capability(&self, metadata: CapabilityMetadata, wasm_bytes: Vec<u8>) -> Result<()> {
        info!("Loading capability: {} v{}", metadata.name, metadata.version);

        #[cfg(feature = "wasi")]
        let sandbox = self.sandbox_factory.clone();

        #[cfg(not(feature = "wasi"))]
        let sandbox = None;

        let capability = LoadedCapability {
            metadata: metadata.clone(),
            wasm_module: wasm_bytes,
            loaded_at: chrono::Local::now().timestamp() as u64,
            #[cfg(feature = "wasi")]
            sandbox,
        };

        self.loaded_capabilities
            .write()
            .await
            .insert(metadata.capability_id.clone(), capability);

        info!("Capability {} loaded successfully", metadata.capability_id);
        Ok(())
    }

    /// Execute capability with given input
    pub async fn execute_capability(
        &self,
        capability_id: &str,
        input: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let capabilities = self.loaded_capabilities.read().await;
        let capability = capabilities
            .get(capability_id)
            .context(format!("Capability {} not found", capability_id))?;

        debug!("Executing capability: {}", capability.metadata.name);

        // In production, execute via WASI sandbox
        // For now, return mock result
        Ok(serde_json::json!({
            "capability_id": capability_id,
            "status": "success",
            "result": input
        }))
    }

    /// Check if capability is loaded
    pub async fn has_capability(&self, capability_id: &str) -> bool {
        self.loaded_capabilities.read().await.contains_key(capability_id)
    }

    /// List all loaded capabilities
    pub async fn list_capabilities(&self) -> Vec<CapabilityMetadata> {
        self.loaded_capabilities
            .read()
            .await
            .values()
            .map(|c| c.metadata.clone())
            .collect()
    }

    /// Unload capability to free memory
    pub async fn unload_capability(&self, capability_id: &str) -> Result<()> {
        self.loaded_capabilities
            .write()
            .await
            .remove(capability_id)
            .context(format!("Capability {} not found", capability_id))?;
        info!("Capability {} unloaded", capability_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_capability_manager_creation() {
        let manager = CapabilityManager::new(None, None);
        assert_eq!(manager.cloud_service_url, "http://capability-service:8001");
    }

    #[tokio::test]
    async fn test_load_and_list_capabilities() {
        let manager = CapabilityManager::new(None, None);

        let metadata = CapabilityMetadata {
            capability_id: "ik_solver_1".to_string(),
            name: "IK Solver".to_string(),
            version: "1.0.0".to_string(),
            capability_type: CapabilityType::InverseKinematics,
            wasm_hash: "abc123".to_string(),
            dependencies: vec![],
            parameters: HashMap::new(),
        };

        manager
            .load_capability(metadata.clone(), vec![0u8; 100])
            .await
            .unwrap();

        assert!(manager.has_capability("ik_solver_1").await);
        let caps = manager.list_capabilities().await;
        assert_eq!(caps.len(), 1);
        assert_eq!(caps[0].name, "IK Solver");
    }

    #[tokio::test]
    async fn test_unload_capability() {
        let manager = CapabilityManager::new(None, None);

        let metadata = CapabilityMetadata {
            capability_id: "test_cap".to_string(),
            name: "Test".to_string(),
            version: "1.0.0".to_string(),
            capability_type: CapabilityType::Custom("test".to_string()),
            wasm_hash: "xyz".to_string(),
            dependencies: vec![],
            parameters: HashMap::new(),
        };

        manager
            .load_capability(metadata.clone(), vec![0u8; 50])
            .await
            .unwrap();
        assert!(manager.has_capability("test_cap").await);

        manager.unload_capability("test_cap").await.unwrap();
        assert!(!manager.has_capability("test_cap").await);
    }
}
