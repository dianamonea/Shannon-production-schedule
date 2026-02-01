/// Low-Latency Interrupt Handler
/// 
/// Provides fast-path processing for sensor emergency signals that bypass LLM processing.
/// Implements real-time interrupt handling with priority levels.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};
use tokio::time::Instant;
use tracing::{debug, error, info, warn};

use crate::ros_bridge::{RobotControlCommand, ControlCommandType};

/// Interrupt priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum InterruptPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Sensor-based interrupt signal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptSignal {
    pub signal_type: InterruptType,
    pub priority: InterruptPriority,
    pub source: String,                    // sensor or component ID
    pub timestamp: u64,
    pub payload: serde_json::Value,       // sensor data
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterruptType {
    EmergencyStop,                        // E-stop button pressed
    CollisionDetected,                    // Force/torque exceeds threshold
    ToolDropped,                          // End effector lost contact
    PowerLow,                             // Battery/power warning
    OverTemperature,                      // Thermal limits exceeded
    ComponentFailure,                     // Motor, sensor, or actuator failure
    SafetyViolation,                      // Workspace boundary exceeded
    Custom(String),
}

/// Interrupt response action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptResponse {
    pub response_type: ResponseType,
    pub control_command: Option<RobotControlCommand>,
    pub notification: Option<String>,
    pub recovery_action: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ResponseType {
    ImmediateStop,
    SafetyPosition,
    Alert,
    Recovery,
}

/// Fast-path rule for interrupt handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptRule {
    pub rule_id: String,
    pub trigger_type: InterruptType,
    pub priority_threshold: InterruptPriority,
    pub response: InterruptResponse,
    pub cooldown_ms: u64,
    pub enabled: bool,
}

/// Interrupt handler with configurable rules and fast-path execution
pub struct InterruptHandler {
    rules: Arc<RwLock<Vec<InterruptRule>>>,
    last_execution_times: Arc<RwLock<std::collections::HashMap<String, Instant>>>,
    interrupt_tx: mpsc::Sender<InterruptSignal>,
    interrupt_rx: Arc<RwLock<Option<mpsc::Receiver<InterruptSignal>>>>,
    stats: Arc<RwLock<HandlerStats>>,
}

#[derive(Debug, Clone, Default)]
struct HandlerStats {
    total_signals: u64,
    processed_signals: u64,
    emergency_stops: u64,
    avg_latency_us: u64,
    last_update: u64,
}

impl InterruptHandler {
    /// Create new interrupt handler with buffered channel
    pub fn new(channel_capacity: usize) -> Self {
        let (tx, rx) = mpsc::channel(channel_capacity);

        Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            last_execution_times: Arc::new(RwLock::new(std::collections::HashMap::new())),
            interrupt_tx: tx,
            interrupt_rx: Arc::new(RwLock::new(Some(rx))),
            stats: Arc::new(RwLock::new(HandlerStats::default())),
        }
    }

    /// Register an interrupt rule
    pub async fn register_rule(&self, rule: InterruptRule) -> Result<()> {
        info!("Registering interrupt rule: {}", rule.rule_id);
        self.rules.write().await.push(rule);
        Ok(())
    }

    /// Default safety rules
    pub async fn register_default_rules(&self) -> Result<()> {
        // E-stop: immediate stop, no delays
        self.register_rule(InterruptRule {
            rule_id: "emergency_stop".to_string(),
            trigger_type: InterruptType::EmergencyStop,
            priority_threshold: InterruptPriority::Critical,
            response: InterruptResponse {
                response_type: ResponseType::ImmediateStop,
                control_command: Some(RobotControlCommand {
                    command_type: ControlCommandType::StopMovement,
                    target_pose: None,
                    target_joints: None,
                    gripper_position: None,
                    execution_time: None,
                }),
                notification: Some("EMERGENCY STOP ACTIVATED".to_string()),
                recovery_action: Some("manual_inspection_required".to_string()),
            },
            cooldown_ms: 0,  // No cooldown for E-stop
            enabled: true,
        })
        .await?;

        // Collision: stop + retract
        self.register_rule(InterruptRule {
            rule_id: "collision_response".to_string(),
            trigger_type: InterruptType::CollisionDetected,
            priority_threshold: InterruptPriority::High,
            response: InterruptResponse {
                response_type: ResponseType::SafetyPosition,
                control_command: Some(RobotControlCommand {
                    command_type: ControlCommandType::StopMovement,
                    target_pose: None,
                    target_joints: None,
                    gripper_position: None,
                    execution_time: None,
                }),
                notification: Some("COLLISION DETECTED, robot stopped".to_string()),
                recovery_action: Some("operator_assessment_needed".to_string()),
            },
            cooldown_ms: 100,
            enabled: true,
        })
        .await?;

        // Tool dropped: open gripper + safety position
        self.register_rule(InterruptRule {
            rule_id: "tool_dropped".to_string(),
            trigger_type: InterruptType::ToolDropped,
            priority_threshold: InterruptPriority::High,
            response: InterruptResponse {
                response_type: ResponseType::SafetyPosition,
                control_command: Some(RobotControlCommand {
                    command_type: ControlCommandType::GripperCommand,
                    target_pose: None,
                    target_joints: None,
                    gripper_position: Some(0.0),  // Open gripper
                    execution_time: Some(0.5),
                }),
                notification: Some("TOOL DROPPED, gripper opened".to_string()),
                recovery_action: Some("tool_replacement_required".to_string()),
            },
            cooldown_ms: 50,
            enabled: true,
        })
        .await?;

        info!("Default interrupt rules registered");
        Ok(())
    }

    /// Process interrupt signal in fast-path (synchronous)
    /// Returns control command if matching rule found
    pub async fn handle_interrupt(&self, signal: InterruptSignal) -> Result<Option<InterruptResponse>> {
        let start_time = Instant::now();

        // Update stats
        let mut stats = self.stats.write().await;
        stats.total_signals += 1;

        let rules = self.rules.read().await;
        let cooldown_map = self.last_execution_times.read().await;

        for rule in rules.iter() {
            if !rule.enabled {
                continue;
            }

            // Check if rule matches signal type
            if rule.trigger_type != signal.signal_type {
                continue;
            }

            // Check priority
            if signal.priority < rule.priority_threshold {
                debug!(
                    "Signal priority {:?} below threshold {:?}, skipping rule {}",
                    signal.priority, rule.priority_threshold, rule.rule_id
                );
                continue;
            }

            // Check cooldown
            if let Some(last_execution) = cooldown_map.get(&rule.rule_id) {
                let elapsed = last_execution.elapsed().as_millis() as u64;
                if elapsed < rule.cooldown_ms {
                    warn!(
                        "Rule {} in cooldown ({}/{} ms elapsed)",
                        rule.rule_id, elapsed, rule.cooldown_ms
                    );
                    continue;
                }
            }

            // Rule matched, execute response
            warn!(
                "Interrupt rule '{}' triggered by signal type: {:?}, priority: {:?}",
                rule.rule_id, signal.signal_type, signal.priority
            );

            if signal.signal_type == InterruptType::EmergencyStop {
                stats.emergency_stops += 1;
            }

            stats.processed_signals += 1;

            // Update execution time
            drop(cooldown_map);  // Release read lock
            let mut cooldown_map = self.last_execution_times.write().await;
            cooldown_map.insert(rule.rule_id.clone(), Instant::now());

            // Calculate latency
            let latency_us = start_time.elapsed().as_micros() as u64;
            stats.avg_latency_us = (stats.avg_latency_us + latency_us) / 2;
            stats.last_update = chrono::Local::now().timestamp() as u64;

            return Ok(Some(rule.response.clone()));
        }

        // No matching rule found
        debug!("No matching interrupt rule for signal type: {:?}", signal.signal_type);
        stats.processed_signals += 1;

        Ok(None)
    }

    /// Send interrupt signal through async channel
    pub async fn send_interrupt(&self, signal: InterruptSignal) -> Result<()> {
        self.interrupt_tx.send(signal).await?;
        Ok(())
    }

    /// Get handler statistics
    pub async fn get_stats(&self) -> HandlerStats {
        self.stats.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_interrupt_handler_creation() {
        let handler = InterruptHandler::new(10);
        handler.register_default_rules().await.unwrap();

        let rules = handler.rules.read().await;
        assert!(!rules.is_empty());
    }

    #[tokio::test]
    async fn test_emergency_stop_handling() {
        let handler = InterruptHandler::new(10);
        handler.register_default_rules().await.unwrap();

        let signal = InterruptSignal {
            signal_type: InterruptType::EmergencyStop,
            priority: InterruptPriority::Critical,
            source: "e_stop_button".to_string(),
            timestamp: chrono::Local::now().timestamp() as u64,
            payload: serde_json::json!({}),
        };

        let response = handler.handle_interrupt(signal).await.unwrap();
        assert!(response.is_some());
        let resp = response.unwrap();
        assert_eq!(resp.response_type, ResponseType::ImmediateStop);
    }

    #[tokio::test]
    async fn test_collision_response() {
        let handler = InterruptHandler::new(10);
        handler.register_default_rules().await.unwrap();

        let signal = InterruptSignal {
            signal_type: InterruptType::CollisionDetected,
            priority: InterruptPriority::High,
            source: "force_sensor_01".to_string(),
            timestamp: chrono::Local::now().timestamp() as u64,
            payload: serde_json::json!({
                "force_x": 45.2,
                "force_y": 12.3,
                "force_z": 8.1
            }),
        };

        let response = handler.handle_interrupt(signal).await.unwrap();
        assert!(response.is_some());
        assert_eq!(response.unwrap().response_type, ResponseType::SafetyPosition);
    }

    #[tokio::test]
    async fn test_cooldown() {
        let handler = InterruptHandler::new(10);
        handler.register_default_rules().await.unwrap();

        let signal = InterruptSignal {
            signal_type: InterruptType::CollisionDetected,
            priority: InterruptPriority::High,
            source: "sensor".to_string(),
            timestamp: chrono::Local::now().timestamp() as u64,
            payload: serde_json::json!({}),
        };

        // First call should execute
        let resp1 = handler.handle_interrupt(signal.clone()).await.unwrap();
        assert!(resp1.is_some());

        // Immediate second call should be blocked by cooldown
        let resp2 = handler.handle_interrupt(signal.clone()).await.unwrap();
        assert!(resp2.is_none());

        // After cooldown, should execute again
        tokio::time::sleep(Duration::from_millis(150)).await;
        let resp3 = handler.handle_interrupt(signal).await.unwrap();
        assert!(resp3.is_some());
    }
}
