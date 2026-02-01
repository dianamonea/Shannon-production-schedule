/// ROS2 Bridge and Robot Hardware Integration Module
/// 
/// This module provides integration between Shannon agent-core and ROS2-based physical robot platforms.
/// It enables agents to subscribe to sensor topics and publish control commands.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// ROS Topic subscription configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROSTopicSubscription {
    pub topic_name: String,
    pub message_type: String,
    pub queue_size: usize,
}

/// ROS publisher configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROSPublisher {
    pub topic_name: String,
    pub message_type: String,
    pub queue_size: usize,
}

/// Sensor data from ROS topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    pub topic: String,
    pub timestamp: u64,
    pub data: serde_json::Value,
    pub frame_id: Option<String>,
}

/// Robot state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotState {
    pub robot_id: String,
    pub joint_states: Option<JointStates>,
    pub gripper_state: Option<GripperState>,
    pub tool_state: Option<ToolState>,
    pub end_effector_pose: Option<Pose>,
    pub is_moving: bool,
    pub is_fault: bool,
    pub fault_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointStates {
    pub names: Vec<String>,
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub efforts: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GripperState {
    pub position: f64,      // 0.0 (open) to 1.0 (closed)
    pub effort: f64,        // Current effort
    pub is_moving: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolState {
    pub tool_id: String,
    pub status: String,     // IDLE, WORKING, ERROR
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub rx: f64,            // roll
    pub ry: f64,            // pitch
    pub rz: f64,            // yaw
}

/// MoveIt! motion planning result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionPlan {
    pub trajectory_points: Vec<TrajectoryPoint>,
    pub planning_time: f64,
    pub success: bool,
    pub error_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub positions: Vec<f64>,
    pub velocities: Vec<f64>,
    pub accelerations: Vec<f64>,
    pub time_from_start: f64,
}

/// Control command to be sent to robot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotControlCommand {
    pub command_type: ControlCommandType,
    pub target_pose: Option<Pose>,
    pub target_joints: Option<Vec<f64>>,
    pub gripper_position: Option<f64>,
    pub execution_time: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum ControlCommandType {
    MoveToCartesian,
    MoveToJoints,
    GripperCommand,
    StopMovement,
    Emergency,
}

/// ROSTool - enables agent to interact with ROS2 ecosystem
pub struct ROSTool {
    enabled: bool,
    ros_master_uri: String,
    subscriptions: Arc<RwLock<HashMap<String, ROSTopicSubscription>>>,
    publishers: Arc<RwLock<HashMap<String, ROSPublisher>>>,
    robot_states: Arc<RwLock<HashMap<String, RobotState>>>,
    emergency_stop_triggered: Arc<RwLock<bool>>,
}

impl ROSTool {
    pub fn new(ros_master_uri: Option<String>) -> Self {
        let uri = ros_master_uri
            .or_else(|| std::env::var("ROS_MASTER_URI").ok())
            .unwrap_or_else(|| "http://localhost:11311".to_string());

        Self {
            enabled: std::env::var("ROS2_ENABLED").unwrap_or_default() == "true",
            ros_master_uri: uri,
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
            publishers: Arc::new(RwLock::new(HashMap::new())),
            robot_states: Arc::new(RwLock::new(HashMap::new())),
            emergency_stop_triggered: Arc::new(RwLock::new(false)),
        }
    }

    pub async fn subscribe_to_topic(&self, subscription: ROSTopicSubscription) -> Result<()> {
        if !self.enabled {
            return Err(anyhow::anyhow!("ROS2 tool not enabled"));
        }

        info!(
            "Subscribing to ROS topic: {} (type: {})",
            subscription.topic_name, subscription.message_type
        );

        // In production, this would use rclrs or zenoh-p2p
        // For now, we simulate the subscription
        self.subscriptions
            .write()
            .await
            .insert(subscription.topic_name.clone(), subscription);

        Ok(())
    }

    pub async fn publish_control_command(&self, command: RobotControlCommand) -> Result<()> {
        if !self.enabled {
            return Err(anyhow::anyhow!("ROS2 tool not enabled"));
        }

        debug!("Publishing control command: {:?}", command);

        // Check if emergency stop is triggered
        if *self.emergency_stop_triggered.read().await {
            warn!("Emergency stop is active, rejecting control command");
            return Err(anyhow::anyhow!("Emergency stop is active"));
        }

        // In production, this would publish to /move_group/goal or equivalent
        // For now, we simulate the publication
        Ok(())
    }

    /// Get current robot state from cached sensor readings
    pub async fn get_robot_state(&self, robot_id: &str) -> Result<Option<RobotState>> {
        Ok(self.robot_states.read().await.get(robot_id).cloned())
    }

    /// Update robot state from sensor readings (called by sensor subscription handler)
    pub async fn update_robot_state(&self, robot_id: String, state: RobotState) {
        self.robot_states.write().await.insert(robot_id, state);
    }

    /// Handle emergency interrupt - bypasses normal LLM processing
    pub async fn trigger_emergency_stop(&self) -> Result<()> {
        warn!("EMERGENCY STOP triggered");
        *self.emergency_stop_triggered.write().await = true;

        // Publish emergency stop to all robots
        // This is a direct hardware command that doesn't go through LLM
        debug!("All robots receiving emergency stop signal");

        Ok(())
    }

    /// Clear emergency stop flag
    pub async fn clear_emergency_stop(&self) -> Result<()> {
        info!("Emergency stop cleared");
        *self.emergency_stop_triggered.write().await = false;
        Ok(())
    }

    pub fn is_emergency_active(&self) -> bool {
        // Non-async check for fast path
        futures::executor::block_on(async {
            *self.emergency_stop_triggered.read().await
        })
    }

    /// Request motion plan from MoveIt!
    pub async fn plan_trajectory(
        &self,
        target_pose: Pose,
        planning_time_limit: f64,
    ) -> Result<MotionPlan> {
        if !self.enabled {
            return Err(anyhow::anyhow!("ROS2 tool not enabled"));
        }

        debug!(
            "Planning trajectory to pose: ({}, {}, {})",
            target_pose.x, target_pose.y, target_pose.z
        );

        // In production, this would call MoveIt! planning service
        // For now, return a mock plan
        Ok(MotionPlan {
            trajectory_points: vec![],
            planning_time: 0.5,
            success: true,
            error_code: None,
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rostool_creation() {
        let tool = ROSTool::new(Some("http://localhost:11311".to_string()));
        assert_eq!(tool.ros_master_uri, "http://localhost:11311");
    }

    #[tokio::test]
    async fn test_robot_state_update() {
        let tool = ROSTool::new(None);
        let state = RobotState {
            robot_id: "ur10_01".to_string(),
            joint_states: None,
            gripper_state: None,
            tool_state: None,
            end_effector_pose: None,
            is_moving: false,
            is_fault: false,
            fault_code: None,
        };

        tool.update_robot_state("ur10_01".to_string(), state.clone()).await;
        let retrieved = tool.get_robot_state("ur10_01").await.unwrap();
        assert_eq!(retrieved.unwrap().robot_id, "ur10_01");
    }

    #[tokio::test]
    async fn test_emergency_stop() {
        let tool = ROSTool::new(None);
        tool.trigger_emergency_stop().await.unwrap();
        assert!(tool.is_emergency_active());
        tool.clear_emergency_stop().await.unwrap();
        assert!(!tool.is_emergency_active());
    }
}
