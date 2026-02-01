use super::primitives::*;
use std::time::Duration;

/// Generate a simple linear Cartesian trajectory between two poses.
pub fn generate_linear_trajectory(start: Pose, goal: Pose, duration: Duration, steps: usize) -> TrajectoryAction {
    let steps = steps.max(2);
    let mut waypoints = Vec::with_capacity(steps);

    for i in 0..steps {
        let t = i as f64 / (steps - 1) as f64;
        let pos = [
            lerp(start.position[0], goal.position[0], t),
            lerp(start.position[1], goal.position[1], t),
            lerp(start.position[2], goal.position[2], t),
        ];

        let pose = Pose::new(pos, slerp_quat(start.orientation, goal.orientation, t));
        waypoints.push(Waypoint {
            pose,
            timestamp: t * duration.as_secs_f64(),
        });
    }

    TrajectoryAction {
        waypoints,
        total_duration: duration,
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn slerp_quat(a: Quaternion, b: Quaternion, t: f64) -> Quaternion {
    // Simplified slerp (normalized lerp) for small angles
    let w = lerp(a.w, b.w, t);
    let x = lerp(a.x, b.x, t);
    let y = lerp(a.y, b.y, t);
    let z = lerp(a.z, b.z, t);
    Quaternion::new(w, x, y, z).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_linear_trajectory() {
        let start = Pose::new([0.0, 0.0, 0.0], Quaternion::identity());
        let goal = Pose::new([1.0, 0.0, 0.0], Quaternion::identity());
        let traj = generate_linear_trajectory(start, goal, Duration::from_secs(2), 5);
        assert_eq!(traj.waypoints.len(), 5);
        assert_eq!(traj.waypoints.first().unwrap().pose.position[0], 0.0);
        assert_eq!(traj.waypoints.last().unwrap().pose.position[0], 1.0);
    }
}
