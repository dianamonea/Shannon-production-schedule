package worldmodel

import (
	"encoding/json"
	"time"
)

// ObservationFromMap converts a generic map into SensorObservation.
// Expected keys: object_id, x, y, z, qw, qx, qy, qz.
func ObservationFromMap(m map[string]interface{}) (SensorObservation, bool) {
	if m == nil {
		return SensorObservation{}, false
	}

	objectID, ok := m["object_id"].(string)
	if !ok || objectID == "" {
		return SensorObservation{}, false
	}

	pose := Pose{
		X:  parseFloat(m["x"], 0),
		Y:  parseFloat(m["y"], 0),
		Z:  parseFloat(m["z"], 0),
		Qw: parseFloat(m["qw"], 1),
		Qx: parseFloat(m["qx"], 0),
		Qy: parseFloat(m["qy"], 0),
		Qz: parseFloat(m["qz"], 0),
	}

	cov := [3][3]float64{{0.1, 0, 0}, {0, 0.1, 0}, {0, 0, 0.1}}

	return SensorObservation{
		ObjectID:   objectID,
		Pose:       pose,
		Covariance: cov,
		ObservedAt: time.Now(),
	}, true
}

// ObservationsFromMaps converts a list of generic maps into observations.
func ObservationsFromMaps(items []map[string]interface{}) []SensorObservation {
	obs := make([]SensorObservation, 0, len(items))
	for _, item := range items {
		if o, ok := ObservationFromMap(item); ok {
			obs = append(obs, o)
		}
	}
	return obs
}

// ObservationsFromJSON parses JSON array of semantic observations
// Expected fields: object_id, pose:{x,y,z,qw,qx,qy,qz}
func ObservationsFromJSON(payload []byte) []SensorObservation {
	var raw []map[string]interface{}
	if err := json.Unmarshal(payload, &raw); err != nil {
		return nil
	}

	items := make([]map[string]interface{}, 0, len(raw))
	for _, entry := range raw {
		poseMap := map[string]interface{}{}
		if pose, ok := entry["pose"].(map[string]interface{}); ok {
			poseMap = pose
		}

		pos := parseFloatSlice(poseMap["position"], 3)
		quat := parseFloatSlice(poseMap["orientation"], 4)
		item := map[string]interface{}{
			"object_id": entry["object_id"],
			"x":         pos[0],
			"y":         pos[1],
			"z":         pos[2],
			"qw":        quat[0],
			"qx":        quat[1],
			"qy":        quat[2],
			"qz":        quat[3],
		}
		items = append(items, item)
	}

	return ObservationsFromMaps(items)
}

// UpdateFromJSON parses JSON observations and updates world model.
// Returns number of observations applied.
func (wm *WorldModel) UpdateFromJSON(payload []byte) int {
	observations := ObservationsFromJSON(payload)
	wm.UpdateObjects(observations)
	return len(observations)
}

func parseFloatSlice(value interface{}, size int) []float64 {
	result := make([]float64, size)
	arr, ok := value.([]interface{})
	if !ok {
		return result
	}
	for i := 0; i < size && i < len(arr); i++ {
		result[i] = parseFloat(arr[i], 0)
	}
	return result
}

func parseFloat(value interface{}, fallback float64) float64 {
	switch v := value.(type) {
	case float64:
		return v
	case float32:
		return float64(v)
	case int:
		return float64(v)
	case int64:
		return float64(v)
	default:
		return fallback
	}
}
