package worldmodel

import (
	"encoding/json"
	"time"
)

// SemanticObject represents a semantic detection result for world model ingestion.
type SemanticObject struct {
	ObjectID   string
	Label      string
	Pose       Pose
	Covariance [3][3]float64
}

// UpdateFromSemantic ingests semantic objects into the world model.
// If a label indicates a support surface, it will register a SurfacePatch.
func (wm *WorldModel) UpdateFromSemantic(objects []SemanticObject) {
	for _, obj := range objects {
		obs := SensorObservation{
			ObjectID:   obj.ObjectID,
			Pose:       obj.Pose,
			Covariance: obj.Covariance,
			ObservedAt: time.Now(),
		}
		wm.UpdateObject(obs)

		if wm.grid != nil {
			_ = wm.grid.UpdateCell(obj.Pose.X, obj.Pose.Y, 1.0)
		}

		if isSurfaceLabel(obj.Label) {
			friction := surfaceFriction(obj.Label)
			wm.AddOrUpdateSurface(&SurfacePatch{
				SurfaceID: obj.ObjectID,
				Center:    obj.Pose,
				Normal:    [3]float64{0, 0, 1},
				Area:      1.0,
				Friction:  friction,
				UpdatedAt: time.Now(),
			})
		}
	}
}

// SemanticObjectsFromJSON parses JSON array of semantic objects.
// Expected fields: object_id, label, pose:{position:[x,y,z], orientation:[qw,qx,qy,qz]}
func SemanticObjectsFromJSON(payload []byte) []SemanticObject {
	var raw []map[string]interface{}
	if err := json.Unmarshal(payload, &raw); err != nil {
		return nil
	}
	objs := make([]SemanticObject, 0, len(raw))
	for _, entry := range raw {
		objectID, _ := entry["object_id"].(string)
		label, _ := entry["label"].(string)
		poseMap, _ := entry["pose"].(map[string]interface{})
		pos := parseFloatSlice(poseMap["position"], 3)
		quat := parseFloatSlice(poseMap["orientation"], 4)
		objs = append(objs, SemanticObject{
			ObjectID: objectID,
			Label:    label,
			Pose: Pose{
				X: pos[0], Y: pos[1], Z: pos[2],
				Qw: quat[0], Qx: quat[1], Qy: quat[2], Qz: quat[3],
			},
			Covariance: [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}},
		})
	}
	return objs
}

// UpdateFromSemanticJSON parses JSON semantic objects and updates world model.
// Returns number of objects ingested.
func (wm *WorldModel) UpdateFromSemanticJSON(payload []byte) int {
	objs := SemanticObjectsFromJSON(payload)
	wm.UpdateFromSemantic(objs)
	return len(objs)
}

func isSurfaceLabel(label string) bool {
	switch label {
	case "table", "surface", "bench", "fixture":
		return true
	default:
		return false
	}
}

func surfaceFriction(label string) float64 {
	switch label {
	case "fixture":
		return 0.8
	case "bench":
		return 0.7
	case "table":
		return 0.6
	default:
		return 0.5
	}
}
