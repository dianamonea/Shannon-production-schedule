use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection2D {
    pub label: String,
    pub confidence: f64,
    pub bbox: [f64; 4], // x, y, w, h
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pose6D {
    pub position: [f64; 3],
    pub orientation: [f64; 4], // qw, qx, qy, qz
    pub covariance: [f64; 9],   // 3x3 position covariance
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub object_id: String,
    pub label: String,
    pub confidence: f64,
    pub pose: Pose6D,
    pub last_seen: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneNode {
    pub node_id: String,
    pub label: String,
    pub attributes: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneEdge {
    pub from: String,
    pub to: String,
    pub relation: String, // on, inside, near, attached_to
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneGraph {
    pub nodes: HashMap<String, SceneNode>,
    pub edges: Vec<SceneEdge>,
    pub updated_at: SystemTime,
}

impl SceneGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            updated_at: SystemTime::now(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticPerception {
    pub objects: HashMap<String, DetectedObject>,
    pub scene_graph: SceneGraph,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticObservation {
    pub object_id: String,
    pub label: String,
    pub confidence: f64,
    pub pose: Pose6D,
}

impl SemanticPerception {
    pub fn new() -> Self {
        Self {
            objects: HashMap::new(),
            scene_graph: SceneGraph::new(),
        }
    }

    /// Update objects with new detections and poses (placeholder for real models)
    pub fn update_objects(&mut self, detections: Vec<Detection2D>, poses: Vec<Pose6D>) {
        for (idx, det) in detections.iter().enumerate() {
            let pose = poses.get(idx).cloned().unwrap_or(Pose6D {
                position: [0.0, 0.0, 0.0],
                orientation: [1.0, 0.0, 0.0, 0.0],
                covariance: [0.1; 9],
            });

            let obj_id = format!("{}_{}", det.label, idx);
            let obj = DetectedObject {
                object_id: obj_id.clone(),
                label: det.label.clone(),
                confidence: det.confidence,
                pose,
                last_seen: SystemTime::now(),
            };
            self.objects.insert(obj_id, obj);
        }

        self.update_scene_graph();
    }

    /// Update scene graph relations using simple spatial heuristics
    pub fn update_scene_graph(&mut self) {
        self.scene_graph.nodes.clear();
        self.scene_graph.edges.clear();

        for (id, obj) in &self.objects {
            self.scene_graph.nodes.insert(
                id.clone(),
                SceneNode {
                    node_id: id.clone(),
                    label: obj.label.clone(),
                    attributes: HashMap::new(),
                },
            );
        }

        let ids: Vec<String> = self.objects.keys().cloned().collect();
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let a = &self.objects[&ids[i]];
                let b = &self.objects[&ids[j]];
                let dist = distance3d(a.pose.position, b.pose.position);
                if dist < 0.2 {
                    self.scene_graph.edges.push(SceneEdge {
                        from: a.object_id.clone(),
                        to: b.object_id.clone(),
                        relation: "near".to_string(),
                        confidence: 0.7,
                    });
                }
            }
        }

        self.scene_graph.updated_at = SystemTime::now();
    }

    /// Export semantic observations for downstream world model updates
    pub fn to_observations(&self) -> Vec<SemanticObservation> {
        self.objects
            .values()
            .map(|obj| SemanticObservation {
                object_id: obj.object_id.clone(),
                label: obj.label.clone(),
                confidence: obj.confidence,
                pose: obj.pose.clone(),
            })
            .collect()
    }

    /// Serialize observations to JSON for cross-language ingestion
    pub fn observations_to_json(&self) -> Result<String, String> {
        let observations = self.to_observations();
        serde_json::to_string(&observations).map_err(|e| e.to_string())
    }
}

fn distance3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_update() {
        let mut sem = SemanticPerception::new();

        let detections = vec![Detection2D {
            label: "bolt".to_string(),
            confidence: 0.9,
            bbox: [0.1, 0.1, 0.2, 0.2],
        }];

        let poses = vec![Pose6D {
            position: [0.0, 0.0, 0.0],
            orientation: [1.0, 0.0, 0.0, 0.0],
            covariance: [0.05; 9],
        }];

        sem.update_objects(detections, poses);
        assert_eq!(sem.objects.len(), 1);
        assert_eq!(sem.scene_graph.nodes.len(), 1);
    }

    #[test]
    fn test_export_observations() {
        let mut sem = SemanticPerception::new();
        sem.update_objects(
            vec![Detection2D {
                label: "part".to_string(),
                confidence: 0.8,
                bbox: [0.0, 0.0, 0.1, 0.1],
            }],
            vec![Pose6D {
                position: [0.2, 0.1, 0.0],
                orientation: [1.0, 0.0, 0.0, 0.0],
                covariance: [0.1; 9],
            }],
        );

        let obs = sem.to_observations();
        assert_eq!(obs.len(), 1);
        assert!(obs[0].object_id.contains("part"));
        assert_eq!(obs[0].label, "part");
    }

    #[test]
    fn test_observations_to_json() {
        let mut sem = SemanticPerception::new();
        sem.update_objects(
            vec![Detection2D {
                label: "fixture".to_string(),
                confidence: 0.9,
                bbox: [0.0, 0.0, 0.1, 0.1],
            }],
            vec![Pose6D {
                position: [0.0, 0.0, 0.0],
                orientation: [1.0, 0.0, 0.0, 0.0],
                covariance: [0.1; 9],
            }],
        );

        let json = sem.observations_to_json().unwrap();
        assert!(json.contains("fixture"));
    }
}
