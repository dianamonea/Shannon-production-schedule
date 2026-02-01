use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionPrimitive {
    MoveToCartesianPose(CartesianMove),
    MoveToJointState(JointMove),
    GraspObject(GraspAction),
    ReleaseObject(ReleaseAction),
    ApplyForce(ForceControl),
    FollowTrajectory(TrajectoryAction),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CartesianMove {
    pub target_pose: Pose,
    pub velocity_limit: Option<f64>, // m/s
    pub acceleration_limit: Option<f64>,
    pub motion_profile: MotionProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pose {
    pub position: [f64; 3],
    pub orientation: Quaternion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Quaternion {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointMove {
    pub target_joints: Vec<f64>, // rad
    pub velocity_limits: Option<Vec<f64>>,
    pub motion_profile: MotionProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MotionProfile {
    Trapezoidal, // 梯形速度曲线
    SCurve,      // S曲线（平滑）
    Linear,      // 恒速
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraspAction {
    pub grasp_pose: Pose,
    pub pre_grasp_offset: f64, // 预抓取偏移（m）
    pub gripper_width: f64,    // 夹爪开口（m）
    pub grasp_force: f64,      // 抓取力（N）
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAction {
    pub release_pose: Pose,
    pub post_release_offset: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceControl {
    pub target_force: [f64; 3],
    pub compliance_axes: [bool; 6], // xyz, rpy
    pub duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryAction {
    pub waypoints: Vec<Waypoint>,
    pub total_duration: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Waypoint {
    pub pose: Pose,
    pub timestamp: f64, // 相对起始时间（秒）
}

impl Pose {
    pub fn new(position: [f64; 3], orientation: Quaternion) -> Self {
        Self {
            position,
            orientation,
        }
    }

    pub fn identity() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            orientation: Quaternion::identity(),
        }
    }
}

impl Quaternion {
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn normalize(&self) -> Self {
        let mag = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if mag < 1e-6 {
            return Self::identity();
        }
        Self {
            w: self.w / mag,
            x: self.x / mag,
            y: self.y / mag,
            z: self.z / mag,
        }
    }
}
