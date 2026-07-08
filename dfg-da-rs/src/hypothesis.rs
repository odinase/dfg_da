use std::collections::BTreeSet;

#[derive(Debug, Clone)]
pub struct Hypothesis {
    log_weight: f64,
    tracks: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Hypotheses {
    hypotheseses: Vec<Hypothesis>,
}

impl Hypotheses {
    pub fn all_tracks(&self) -> BTreeSet<usize> {
        self.hypotheseses
            .iter()
            .flat_map(|h| h.tracks.iter())
            .copied()
            .collect()
    }
}
