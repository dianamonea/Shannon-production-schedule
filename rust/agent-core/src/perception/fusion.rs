use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use super::semantic::*;
use super::sensor_types::*;

pub struct SensorFusion {
    sensor_buffers: Arc<Mutex<HashMap<String, Vec<SensorReading>>>>,
    time_sync_tolerance: Duration,
    sensor_transforms: Arc<Mutex<HashMap<String, [f64; 3]>>>,
}

#[derive(Debug, Clone)]
pub struct GraspPose {
    pub position: [f64; 3],    // xyz (m)
    pub orientation: [f64; 4], // Quaternion (w, x, y, z)
    pub confidence: f64,       // 0-1
}

impl SensorFusion {
    pub fn new() -> Self {
        Self {
            sensor_buffers: Arc::new(Mutex::new(HashMap::new())),
            time_sync_tolerance: Duration::from_millis(10), // 10ms同步窗口
            sensor_transforms: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn set_time_sync_tolerance(&mut self, tolerance: Duration) {
        self.time_sync_tolerance = tolerance;
    }

    pub fn set_sensor_translation(&self, sensor_id: &str, translation: [f64; 3]) {
        let mut transforms = self.sensor_transforms.lock().unwrap();
        transforms.insert(sensor_id.to_string(), translation);
    }

    /// 添加传感器读数到缓冲区
    pub fn add_reading(&self, reading: SensorReading) {
        let mut buffers = self.sensor_buffers.lock().unwrap();
        let buffer = buffers
            .entry(reading.sensor_id.clone())
            .or_insert_with(Vec::new);
        buffer.push(reading);

        // 保留最近1秒的数据
        let cutoff_time = SystemTime::now() - Duration::from_secs(1);
        buffer.retain(|r| r.timestamp > cutoff_time);
    }

    /// 视觉+力觉融合抓取
    pub fn fuse_vision_force_for_grasp(&self) -> Result<GraspPose, String> {
        let buffers = self.sensor_buffers.lock().unwrap();

        // 1. 获取最新的视觉数据
        let camera_readings = buffers
            .get("camera_wrist")
            .ok_or("No camera data")?;
        let latest_image = camera_readings.last().ok_or("No recent image")?;

        // 2. 提取物体位姿（调用视觉模型）
        let object_pose = self.detect_object_pose(latest_image)?;

        // 3. 获取同步的力觉数据
        let force_readings = buffers
            .get("ft_sensor")
            .ok_or("No force/torque data")?;
        let synced_force = self.find_synced_reading(
            latest_image.timestamp,
            force_readings,
            self.time_sync_tolerance,
        )?;

        // 4. 融合：如果检测到接触力，调整抓取姿态
        let adjusted_pose = if let SensorData::ForceTorque(ft) = &synced_force.data {
            let force_magnitude = (ft.force[0].powi(2)
                + ft.force[1].powi(2)
                + ft.force[2].powi(2))
            .sqrt();

            if force_magnitude > 1.0 {
                // 1N阈值
                // 有接触力 → 视觉可能不准确，使用力反馈微调
                self.adjust_pose_by_force(object_pose, ft)
            } else {
                object_pose
            }
        } else {
            object_pose
        };

        Ok(adjusted_pose)
    }

    /// 点云配准（多相机融合）
    pub fn fuse_point_clouds(&self, camera_ids: Vec<String>) -> Result<PointCloudData, String> {
        let buffers = self.sensor_buffers.lock().unwrap();
        let transforms = self.sensor_transforms.lock().unwrap();

        let mut all_points = Vec::new();

        for camera_id in camera_ids {
            let readings = buffers
                .get(&camera_id)
                .ok_or_else(|| format!("No data from {}", camera_id))?;
            let latest = readings
                .last()
                .ok_or_else(|| format!("No recent data from {}", camera_id))?;

            if let SensorData::PointCloud(pc) = &latest.data {
                let offset = transforms.get(&camera_id).cloned().unwrap_or([0.0, 0.0, 0.0]);
                for p in &pc.points {
                    all_points.push(Point3D {
                        x: p.x + offset[0],
                        y: p.y + offset[1],
                        z: p.z + offset[2],
                    });
                }
            }
        }

        Ok(PointCloudData {
            points: all_points,
            colors: None,
        })
    }

    /// 语义场景构建（占位实现）
    pub fn build_semantic_scene(&self) -> Result<SemanticPerception, String> {
        let buffers = self.sensor_buffers.lock().unwrap();

        let camera_readings = buffers
            .get("camera_wrist")
            .ok_or("No camera data")?;
        let latest_image = camera_readings.last().ok_or("No recent image")?;

        // TODO: 用视觉模型生成真实检测与位姿
        let detections = vec![Detection2D {
            label: "part".to_string(),
            confidence: 0.8,
            bbox: [0.1, 0.1, 0.2, 0.2],
        }];

        let poses = vec![Pose6D {
            position: [0.4, 0.2, 0.1],
            orientation: [1.0, 0.0, 0.0, 0.0],
            covariance: [0.05; 9],
        }];

        let mut semantic = SemanticPerception::new();
        semantic.update_objects(detections, poses);

        // Hint to avoid unused variable in future integration
        let _ = latest_image;

        Ok(semantic)
    }

    /// 清空所有缓冲区
    pub fn clear_buffers(&self) {
        let mut buffers = self.sensor_buffers.lock().unwrap();
        buffers.clear();
    }

    /// 获取传感器数据数量
    pub fn get_buffer_size(&self, sensor_id: &str) -> usize {
        let buffers = self.sensor_buffers.lock().unwrap();
        buffers.get(sensor_id).map(|b| b.len()).unwrap_or(0)
    }

    // 辅助函数

    fn find_synced_reading<'a>(
        &self,
        target_time: SystemTime,
        readings: &'a [SensorReading],
        tolerance: Duration,
    ) -> Result<&'a SensorReading, String> {
        readings
            .iter()
            .filter(|r| {
                let diff = r
                    .timestamp
                    .duration_since(target_time)
                    .or_else(|_| target_time.duration_since(r.timestamp))
                    .unwrap_or(Duration::from_secs(100));
                diff < tolerance
            })
            .min_by_key(|r| {
                r.timestamp
                    .duration_since(target_time)
                    .or_else(|_| target_time.duration_since(r.timestamp))
                    .unwrap_or(Duration::from_secs(100))
            })
            .ok_or_else(|| "No synced reading found".to_string())
    }

    fn detect_object_pose(&self, _image_reading: &SensorReading) -> Result<GraspPose, String> {
        // TODO: 调用YOLO/Mask R-CNN检测物体边界框
        // TODO: 调用6D姿态估计网络（例如：DenseFusion, PVN3D）

        // 占位实现
        Ok(GraspPose {
            position: [0.5, 0.2, 0.1],
            orientation: [1.0, 0.0, 0.0, 0.0],
            confidence: 0.85,
        })
    }

    fn adjust_pose_by_force(&self, pose: GraspPose, ft: &ForceTorqueData) -> GraspPose {
        // 简化：根据力方向微调位置
        let force_norm =
            (ft.force[0].powi(2) + ft.force[1].powi(2) + ft.force[2].powi(2)).sqrt();

        if force_norm < 0.001 {
            return pose;
        }

        let force_unit = [
            ft.force[0] / force_norm,
            ft.force[1] / force_norm,
            ft.force[2] / force_norm,
        ];

        // 沿力方向偏移5mm
        let offset = 0.005;
        GraspPose {
            position: [
                pose.position[0] + force_unit[0] * offset,
                pose.position[1] + force_unit[1] * offset,
                pose.position[2] + force_unit[2] * offset,
            ],
            orientation: pose.orientation,
            confidence: pose.confidence * 0.9, // 力反馈调整后置信度略降
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_reading() {
        let fusion = SensorFusion::new();

        let reading = SensorReading {
            sensor_id: "test_sensor".to_string(),
            sensor_type: SensorType::Camera,
            timestamp: SystemTime::now(),
            data: SensorData::Image(ImageData {
                width: 640,
                height: 480,
                channels: 3,
                encoding: "rgb8".to_string(),
                data: vec![0; 640 * 480 * 3],
            }),
        };

        fusion.add_reading(reading);

        assert_eq!(fusion.get_buffer_size("test_sensor"), 1);
    }

    #[test]
    fn test_buffer_cleanup() {
        let fusion = SensorFusion::new();

        // 添加旧数据（不应保留）
        let old_reading = SensorReading {
            sensor_id: "test_sensor".to_string(),
            sensor_type: SensorType::Camera,
            timestamp: SystemTime::now() - Duration::from_secs(2),
            data: SensorData::Image(ImageData {
                width: 640,
                height: 480,
                channels: 3,
                encoding: "rgb8".to_string(),
                data: vec![0; 640 * 480 * 3],
            }),
        };

        fusion.add_reading(old_reading);

        // 添加新数据（应保留）
        let new_reading = SensorReading {
            sensor_id: "test_sensor".to_string(),
            sensor_type: SensorType::Camera,
            timestamp: SystemTime::now(),
            data: SensorData::Image(ImageData {
                width: 640,
                height: 480,
                channels: 3,
                encoding: "rgb8".to_string(),
                data: vec![0; 640 * 480 * 3],
            }),
        };

        fusion.add_reading(new_reading);

        // 旧数据应被清理
        assert_eq!(fusion.get_buffer_size("test_sensor"), 1);
    }

    #[test]
    fn test_clear_buffers() {
        let fusion = SensorFusion::new();

        let reading = SensorReading {
            sensor_id: "test_sensor".to_string(),
            sensor_type: SensorType::Camera,
            timestamp: SystemTime::now(),
            data: SensorData::Image(ImageData {
                width: 640,
                height: 480,
                channels: 3,
                encoding: "rgb8".to_string(),
                data: vec![0; 640 * 480 * 3],
            }),
        };

        fusion.add_reading(reading);
        assert_eq!(fusion.get_buffer_size("test_sensor"), 1);

        fusion.clear_buffers();
        assert_eq!(fusion.get_buffer_size("test_sensor"), 0);
    }

    #[test]
    fn test_build_semantic_scene() {
        let fusion = SensorFusion::new();

        let reading = SensorReading {
            sensor_id: "camera_wrist".to_string(),
            sensor_type: SensorType::Camera,
            timestamp: SystemTime::now(),
            data: SensorData::Image(ImageData {
                width: 640,
                height: 480,
                channels: 3,
                encoding: "rgb8".to_string(),
                data: vec![0; 640 * 480 * 3],
            }),
        };

        fusion.add_reading(reading);

        let semantic = fusion.build_semantic_scene().unwrap();
        assert_eq!(semantic.objects.len(), 1);
        assert_eq!(semantic.scene_graph.nodes.len(), 1);
    }

    #[test]
    fn test_point_cloud_transform() {
        let fusion = SensorFusion::new();
        fusion.set_sensor_translation("camera_1", [1.0, 0.0, 0.0]);

        let reading = SensorReading {
            sensor_id: "camera_1".to_string(),
            sensor_type: SensorType::LiDAR,
            timestamp: SystemTime::now(),
            data: SensorData::PointCloud(PointCloudData {
                points: vec![Point3D { x: 0.0, y: 0.0, z: 0.0 }],
                colors: None,
            }),
        };

        fusion.add_reading(reading);
        let pc = fusion.fuse_point_clouds(vec!["camera_1".to_string()]).unwrap();
        assert_eq!(pc.points[0].x, 1.0);
    }
}
