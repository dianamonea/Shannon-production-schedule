use super::primitives::*;

#[derive(Debug, Clone)]
pub struct JointLimit {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone)]
pub struct WorkspaceBounds {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct MotionConstraints {
    pub joint_limits: Option<Vec<JointLimit>>,
    pub workspace_bounds: Option<WorkspaceBounds>,
    pub max_velocity: Option<f64>,
    pub max_acceleration: Option<f64>,
    pub max_force: Option<f64>,
    pub max_gripper_width: Option<f64>,
}

impl MotionConstraints {
    pub fn new() -> Self {
        Self {
            joint_limits: None,
            workspace_bounds: None,
            max_velocity: None,
            max_acceleration: None,
            max_force: None,
            max_gripper_width: None,
        }
    }
}

pub fn validate_primitive(primitive: &ActionPrimitive, constraints: &MotionConstraints) -> Result<(), String> {
    match primitive {
        ActionPrimitive::MoveToCartesianPose(move_cmd) => {
            if let Some(bounds) = &constraints.workspace_bounds {
                let p = move_cmd.target_pose.position;
                if p[0] < bounds.min[0] || p[0] > bounds.max[0]
                    || p[1] < bounds.min[1] || p[1] > bounds.max[1]
                    || p[2] < bounds.min[2] || p[2] > bounds.max[2]
                {
                    return Err("Cartesian target out of workspace bounds".to_string());
                }
            }
            if let Some(max_v) = constraints.max_velocity {
                if let Some(v) = move_cmd.velocity_limit {
                    if v > max_v {
                        return Err("Cartesian velocity exceeds limit".to_string());
                    }
                }
            }
            if let Some(max_a) = constraints.max_acceleration {
                if let Some(a) = move_cmd.acceleration_limit {
                    if a > max_a {
                        return Err("Cartesian acceleration exceeds limit".to_string());
                    }
                }
            }
        }
        ActionPrimitive::MoveToJointState(joint_cmd) => {
            if let Some(limits) = &constraints.joint_limits {
                if limits.len() != joint_cmd.target_joints.len() {
                    return Err("Joint limits size mismatch".to_string());
                }
                for (idx, target) in joint_cmd.target_joints.iter().enumerate() {
                    let limit = &limits[idx];
                    if *target < limit.min || *target > limit.max {
                        return Err(format!("Joint {} out of limits", idx));
                    }
                }
            }
            if let Some(max_v) = constraints.max_velocity {
                if let Some(v_limits) = &joint_cmd.velocity_limits {
                    for v in v_limits {
                        if *v > max_v {
                            return Err("Joint velocity exceeds limit".to_string());
                        }
                    }
                }
            }
        }
        ActionPrimitive::GraspObject(grasp) => {
            if let Some(max_w) = constraints.max_gripper_width {
                if grasp.gripper_width > max_w {
                    return Err("Gripper width exceeds limit".to_string());
                }
            }
            if let Some(max_f) = constraints.max_force {
                if grasp.grasp_force > max_f {
                    return Err("Grasp force exceeds limit".to_string());
                }
            }
        }
        ActionPrimitive::ApplyForce(force_cmd) => {
            if let Some(max_f) = constraints.max_force {
                let mag = (force_cmd.target_force[0].powi(2)
                    + force_cmd.target_force[1].powi(2)
                    + force_cmd.target_force[2].powi(2))
                    .sqrt();
                if mag > max_f {
                    return Err("Force command exceeds limit".to_string());
                }
            }
        }
        ActionPrimitive::ReleaseObject(_) => {}
        ActionPrimitive::FollowTrajectory(traj) => {
            if let Some(bounds) = &constraints.workspace_bounds {
                for wp in &traj.waypoints {
                    let p = wp.pose.position;
                    if p[0] < bounds.min[0] || p[0] > bounds.max[0]
                        || p[1] < bounds.min[1] || p[1] > bounds.max[1]
                        || p[2] < bounds.min[2] || p[2] > bounds.max[2]
                    {
                        return Err("Trajectory waypoint out of bounds".to_string());
                    }
                }
            }
            if !traj.waypoints.is_empty() {
                let mut last_ts = -1.0;
                for wp in &traj.waypoints {
                    if wp.timestamp < last_ts {
                        return Err("Trajectory timestamps must be non-decreasing".to_string());
                    }
                    last_ts = wp.timestamp;
                }
                if last_ts > traj.total_duration.as_secs_f64() {
                    return Err("Trajectory exceeds total duration".to_string());
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_cartesian_bounds() {
        let constraints = MotionConstraints {
            workspace_bounds: Some(WorkspaceBounds {
                min: [0.0, 0.0, 0.0],
                max: [1.0, 1.0, 1.0],
            }),
            max_velocity: Some(0.2),
            max_acceleration: Some(1.0),
            ..MotionConstraints::new()
        };

        let primitive = ActionPrimitive::MoveToCartesianPose(CartesianMove {
            target_pose: Pose::new([0.5, 0.5, 0.5], Quaternion::identity()),
            velocity_limit: Some(0.1),
            acceleration_limit: Some(0.5),
            motion_profile: MotionProfile::SCurve,
        });

        assert!(validate_primitive(&primitive, &constraints).is_ok());
    }

    #[test]
    fn test_validate_joint_limits() {
        let constraints = MotionConstraints {
            joint_limits: Some(vec![
                JointLimit { min: -1.0, max: 1.0 },
                JointLimit { min: -1.0, max: 1.0 },
            ]),
            ..MotionConstraints::new()
        };

        let primitive = ActionPrimitive::MoveToJointState(JointMove {
            target_joints: vec![0.2, 0.9],
            velocity_limits: Some(vec![0.1, 0.1]),
            motion_profile: MotionProfile::Trapezoidal,
        });

        assert!(validate_primitive(&primitive, &constraints).is_ok());
    }

    #[test]
    fn test_validate_force_limit() {
        let constraints = MotionConstraints {
            max_force: Some(10.0),
            ..MotionConstraints::new()
        };

        let primitive = ActionPrimitive::ApplyForce(ForceControl {
            target_force: [3.0, 4.0, 0.0],
            compliance_axes: [true, true, true, false, false, false],
            duration: std::time::Duration::from_secs(1),
        });

        assert!(validate_primitive(&primitive, &constraints).is_ok());
    }

    #[test]
    fn test_validate_trajectory_timestamps() {
        let constraints = MotionConstraints::new();
        let traj = TrajectoryAction {
            waypoints: vec![
                Waypoint { pose: Pose::identity(), timestamp: 0.5 },
                Waypoint { pose: Pose::identity(), timestamp: 0.2 },
            ],
            total_duration: std::time::Duration::from_secs(1),
        };

        let primitive = ActionPrimitive::FollowTrajectory(traj);
        assert!(validate_primitive(&primitive, &constraints).is_err());
    }
}
