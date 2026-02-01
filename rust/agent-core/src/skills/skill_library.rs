use std::collections::HashMap;

use super::constraints::*;
use super::primitives::*;

pub struct SkillLibrary {
    skills: HashMap<String, CompositeSkill>,
    constraints: Option<MotionConstraints>,
}

#[derive(Debug, Clone)]
pub struct CompositeSkill {
    pub skill_id: String,
    pub skill_name: String,
    pub parameters: Vec<SkillParameter>,
    pub primitives: Vec<ActionPrimitive>,
}

#[derive(Debug, Clone)]
pub struct SkillParameter {
    pub name: String,
    pub param_type: String, // "pose", "float", "int"
    pub default_value: Option<String>,
}

impl SkillLibrary {
    pub fn new() -> Self {
        let mut library = Self {
            skills: HashMap::new(),
            constraints: None,
        };

        // 预定义常用技能
        library.register_pick_and_place();
        library.register_inspection();

        library
    }

    fn register_pick_and_place(&mut self) {
        let skill = CompositeSkill {
            skill_id: "pick_and_place".to_string(),
            skill_name: "Pick and Place".to_string(),
            parameters: vec![
                SkillParameter {
                    name: "object_pose".to_string(),
                    param_type: "pose".to_string(),
                    default_value: None,
                },
                SkillParameter {
                    name: "target_pose".to_string(),
                    param_type: "pose".to_string(),
                    default_value: None,
                },
            ],
            primitives: vec![
                // 1. 移动到预抓取位置
                ActionPrimitive::MoveToCartesianPose(CartesianMove {
                    target_pose: Pose::identity(), // 占位符，运行时替换
                    velocity_limit: Some(0.1),
                    acceleration_limit: Some(0.5),
                    motion_profile: MotionProfile::SCurve,
                }),
                // 2. 抓取
                ActionPrimitive::GraspObject(GraspAction {
                    grasp_pose: Pose::identity(),
                    pre_grasp_offset: 0.05,
                    gripper_width: 0.08,
                    grasp_force: 20.0,
                }),
                // 3. 移动到目标位置
                ActionPrimitive::MoveToCartesianPose(CartesianMove {
                    target_pose: Pose::identity(),
                    velocity_limit: Some(0.15),
                    acceleration_limit: Some(0.5),
                    motion_profile: MotionProfile::SCurve,
                }),
                // 4. 释放
                ActionPrimitive::ReleaseObject(ReleaseAction {
                    release_pose: Pose::identity(),
                    post_release_offset: 0.05,
                }),
            ],
        };

        self.skills.insert(skill.skill_id.clone(), skill);
    }

    fn register_inspection(&mut self) {
        let skill = CompositeSkill {
            skill_id: "visual_inspection".to_string(),
            skill_name: "Visual Inspection".to_string(),
            parameters: vec![SkillParameter {
                name: "inspection_pose".to_string(),
                param_type: "pose".to_string(),
                default_value: None,
            }],
            primitives: vec![
                // 移动到检测位置
                ActionPrimitive::MoveToCartesianPose(CartesianMove {
                    target_pose: Pose::identity(),
                    velocity_limit: Some(0.05),
                    acceleration_limit: Some(0.3),
                    motion_profile: MotionProfile::SCurve,
                }),
                // TODO: 添加图像采集原语
            ],
        };

        self.skills.insert(skill.skill_id.clone(), skill);
    }

    pub fn get_skill(&self, skill_id: &str) -> Option<&CompositeSkill> {
        self.skills.get(skill_id)
    }

    pub fn list_skills(&self) -> Vec<String> {
        self.skills.keys().cloned().collect()
    }

    pub fn register_custom_skill(&mut self, skill: CompositeSkill) {
        self.skills.insert(skill.skill_id.clone(), skill);
    }

    pub fn set_constraints(&mut self, constraints: MotionConstraints) {
        self.constraints = Some(constraints);
    }

    pub fn execute_skill(
        &self,
        skill_id: &str,
        params: HashMap<String, String>,
    ) -> Result<(), String> {
        let skill = self
            .get_skill(skill_id)
            .ok_or_else(|| format!("Skill {} not found", skill_id))?;

        // 参数验证
        for param in &skill.parameters {
            if param.default_value.is_none() && !params.contains_key(&param.name) {
                return Err(format!("Missing required parameter: {}", param.name));
            }
        }

        // 参数替换
        let resolved_primitives = self.apply_parameters(&skill.primitives, &params)?;

        // 动作约束验证
        if let Some(constraints) = &self.constraints {
            for primitive in &resolved_primitives {
                validate_primitive(primitive, constraints)?;
            }
        }

        // TODO: 参数替换和执行
        println!(
            "Executing skill: {} with {} primitives",
            skill.skill_name,
            resolved_primitives.len()
        );

        Ok(())
    }

    fn apply_parameters(
        &self,
        primitives: &[ActionPrimitive],
        params: &HashMap<String, String>,
    ) -> Result<Vec<ActionPrimitive>, String> {
        let mut resolved = Vec::with_capacity(primitives.len());

        for primitive in primitives {
            let updated = match primitive {
                ActionPrimitive::MoveToCartesianPose(cmd) => {
                    let mut cmd = cmd.clone();
                    if let Some(pose_str) = params.get("object_pose") {
                        cmd.target_pose = parse_pose_string(pose_str)?;
                    }
                    if let Some(pose_str) = params.get("target_pose") {
                        cmd.target_pose = parse_pose_string(pose_str)?;
                    }
                    if let Some(pose_str) = params.get("inspection_pose") {
                        cmd.target_pose = parse_pose_string(pose_str)?;
                    }
                    ActionPrimitive::MoveToCartesianPose(cmd)
                }
                ActionPrimitive::GraspObject(cmd) => {
                    let mut cmd = cmd.clone();
                    if let Some(pose_str) = params.get("object_pose") {
                        cmd.grasp_pose = parse_pose_string(pose_str)?;
                    }
                    ActionPrimitive::GraspObject(cmd)
                }
                ActionPrimitive::ReleaseObject(cmd) => {
                    let mut cmd = cmd.clone();
                    if let Some(pose_str) = params.get("target_pose") {
                        cmd.release_pose = parse_pose_string(pose_str)?;
                    }
                    ActionPrimitive::ReleaseObject(cmd)
                }
                _ => primitive.clone(),
            };
            resolved.push(updated);
        }

        Ok(resolved)
    }
}

fn parse_pose_string(value: &str) -> Result<Pose, String> {
    let parts: Vec<&str> = value.split(',').collect();
    if parts.len() != 7 {
        return Err("Pose must have 7 values: x,y,z,qw,qx,qy,qz".to_string());
    }

    let vals: Result<Vec<f64>, _> = parts.iter().map(|p| p.trim().parse::<f64>()).collect();
    let vals = vals.map_err(|_| "Pose contains non-numeric values".to_string())?;

    Ok(Pose::new(
        [vals[0], vals[1], vals[2]],
        Quaternion::new(vals[3], vals[4], vals[5], vals[6]).normalize(),
    ))
}

impl Default for SkillLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_library_creation() {
        let library = SkillLibrary::new();
        assert!(library.get_skill("pick_and_place").is_some());
        assert!(library.get_skill("visual_inspection").is_some());
    }

    #[test]
    fn test_list_skills() {
        let library = SkillLibrary::new();
        let skills = library.list_skills();
        assert!(skills.contains(&"pick_and_place".to_string()));
        assert!(skills.contains(&"visual_inspection".to_string()));
    }

    #[test]
    fn test_execute_skill() {
        let library = SkillLibrary::new();

        let mut params = HashMap::new();
        params.insert("object_pose".to_string(), "0,0,0,1,0,0,0".to_string());
        params.insert("target_pose".to_string(), "0.5,0,0,1,0,0,0".to_string());

        let result = library.execute_skill("pick_and_place", params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_missing_parameters() {
        let library = SkillLibrary::new();

        let params = HashMap::new(); // 没有提供参数

        let result = library.execute_skill("pick_and_place", params);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Missing required parameter"));
    }

    #[test]
    fn test_custom_skill_registration() {
        let mut library = SkillLibrary::new();

        let custom_skill = CompositeSkill {
            skill_id: "custom_assembly".to_string(),
            skill_name: "Custom Assembly".to_string(),
            parameters: vec![],
            primitives: vec![],
        };

        library.register_custom_skill(custom_skill);
        assert!(library.get_skill("custom_assembly").is_some());
    }

    #[test]
    fn test_invalid_pose_param() {
        let library = SkillLibrary::new();

        let mut params = HashMap::new();
        params.insert("object_pose".to_string(), "0,0,0,1,0,0".to_string());
        params.insert("target_pose".to_string(), "0.5,0,0,1,0,0,0".to_string());

        let result = library.execute_skill("pick_and_place", params);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Pose must have 7 values"));
    }
}
