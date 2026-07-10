use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct Hypothesis {
    tracks: Vec<usize>,
    log_weight: f64,
}

impl Hypothesis {
    pub fn new(tracks: Vec<usize>, log_weight: f64) -> Self {
        Self {
            tracks,
            log_weight,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Hypotheses {
    hypotheses: Vec<Hypothesis>,
}

impl Hypotheses {
    pub fn new() -> Self {
        Self {hypotheses: Vec::new()}
    }

    pub fn with_hypotheses(hypotheses: Vec<Hypothesis>) -> Self {
        Self {
            hypotheses
        }
    }

    pub fn all_tracks(&self) -> BTreeSet<usize> {
        self.hypotheses
            .iter()
            .flat_map(|h| h.tracks.iter())
            .copied()
            .collect()
    }
}
