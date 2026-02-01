use std::collections::HashMap;

use super::skill_library::SkillLibrary;

#[derive(Debug, Clone)]
pub struct TaskPlan {
    pub goal: String,
    pub skills: Vec<String>,
    pub parameters: HashMap<String, String>,
}

pub struct SkillPlanner {
    library: SkillLibrary,
}

impl SkillPlanner {
    pub fn new(library: SkillLibrary) -> Self {
        Self { library }
    }

    /// Create a simple plan by mapping goal to known skills.
    pub fn plan(&self, goal: &str, params: HashMap<String, String>) -> Result<TaskPlan, String> {
        let skills = match goal {
            "pick_and_place" => vec!["pick_and_place".to_string()],
            "inspect" => vec!["visual_inspection".to_string()],
            "pick_inspect_place" => vec![
                "pick_and_place".to_string(),
                "visual_inspection".to_string(),
            ],
            _ => return Err(format!("Unknown goal: {goal}")),
        };

        for s in &skills {
            if self.library.get_skill(s).is_none() {
                return Err(format!("Skill not found: {s}"));
            }
        }

        Ok(TaskPlan {
            goal: goal.to_string(),
            skills,
            parameters: params,
        })
    }

    pub fn execute_plan(&self, plan: &TaskPlan) -> Result<(), String> {
        for skill_id in &plan.skills {
            self.library.execute_skill(skill_id, plan.parameters.clone())?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skill_planner_plan() {
        let planner = SkillPlanner::new(SkillLibrary::new());
        let plan = planner.plan("pick_and_place", HashMap::new()).unwrap();
        assert_eq!(plan.skills.len(), 1);
    }

    #[test]
    fn test_execute_plan() {
        let planner = SkillPlanner::new(SkillLibrary::new());
        let mut params = HashMap::new();
        params.insert("object_pose".to_string(), "0,0,0,1,0,0,0".to_string());
        params.insert("target_pose".to_string(), "0.5,0,0,1,0,0,0".to_string());
        let plan = planner.plan("pick_and_place", params).unwrap();
        assert!(planner.execute_plan(&plan).is_ok());
    }
}
