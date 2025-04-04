use crate::experience::Experience;
use rand::rng;
use rand::seq::IteratorRandom;
use std::collections::VecDeque;

pub struct ReplayBuffer {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        ReplayBuffer {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, experience: Experience) {
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let mut rng = rng();
        let experiences: Vec<Experience> = self.buffer.iter().cloned().collect();
        experiences
            .iter()
            .choose_multiple(&mut rng, batch_size.min(experiences.len()))
            .into_iter()
            .cloned()
            .collect()
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}
