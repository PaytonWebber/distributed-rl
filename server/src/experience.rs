use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Experience {
    pub state: Vec<f32>,
    pub policy: Vec<f32>,
    pub reward: f32,
}
