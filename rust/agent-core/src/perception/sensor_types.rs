use serde::{Deserialize, Serialize};
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReading {
    pub sensor_id: String,
    pub sensor_type: SensorType,
    pub timestamp: SystemTime,
    pub data: SensorData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorType {
    Camera,
    ForceTorqueSensor,
    LiDAR,
    IMU,
    JointEncoder,
    TactileSensor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensorData {
    Image(ImageData),
    ForceTorque(ForceTorqueData),
    PointCloud(PointCloudData),
    IMU(IMUData),
    JointState(JointStateData),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub encoding: String, // "rgb8", "bgr8", "mono8"
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceTorqueData {
    pub force: [f64; 3],  // Fx, Fy, Fz (N)
    pub torque: [f64; 3], // Tx, Ty, Tz (Nm)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudData {
    pub points: Vec<Point3D>,
    pub colors: Option<Vec<RGB>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RGB {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IMUData {
    pub linear_acceleration: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub orientation: Option<[f64; 4]>, // Quaternion (w, x, y, z)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointStateData {
    pub positions: Vec<f64>,  // rad
    pub velocities: Vec<f64>, // rad/s
    pub efforts: Vec<f64>,    // Nm
}
