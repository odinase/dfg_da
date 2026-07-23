use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone)]
pub struct Hypothesis {
    tracks: Vec<usize>,
    log_weight: f64,
}

impl Hypothesis {
    pub fn new(tracks: Vec<usize>, log_weight: f64) -> Self {
        Self { tracks, log_weight }
    }
    /// In-place reindexing — the workhorse.
    pub fn reindex(&mut self, old2new: &BTreeMap<usize, usize>) {
        for t in &mut self.tracks {
            *t = old2new[t];
        }
    }

    /// Chaining wrapper, kept for call-site ergonomics.
    pub fn into_reindexed(mut self, old2new: &BTreeMap<usize, usize>) -> Self {
        self.reindex(old2new);
        self
    }

    pub fn tracks(&self) -> &[usize] {
        self.tracks.as_slice()
    }

    pub fn probability(&self) -> f64 {
        self.log_weight.exp()
    }
}

fn log_normalize(mut hypotheses: Vec<Hypothesis>) -> Vec<Hypothesis> {
    if hypotheses.is_empty() {
        // No hypotheses to log normalize
        return hypotheses;
    }

    // Sanitze
    for hyp in &mut hypotheses {
        if hyp.log_weight.is_nan() {
            hyp.log_weight = f64::NEG_INFINITY;
        }
    }

    // find max
    let log_weight_max = hypotheses
        .iter()
        .map(|h| h.log_weight)
        .max_by(|lhs, rhs| lhs.partial_cmp(&rhs).unwrap())
        // Guaranteed to at least one element due to if above
        .unwrap();

    let log_exp_neg_max = hypotheses
        .iter()
        .map(|h| (h.log_weight - log_weight_max).exp())
        .sum::<f64>()
        .ln();
    let logsumexp = log_weight_max + log_exp_neg_max;

    for h in &mut hypotheses {
        h.log_weight -= logsumexp;
    }

    hypotheses
}

#[derive(Debug, Clone)]
pub struct Hypotheses {
    hypotheses: Vec<Hypothesis>,
}

impl Hypotheses {
    pub fn new() -> Self {
        Self {
            hypotheses: Vec::new(),
        }
    }

    pub fn num_hypotheses(&self) -> usize {
        self.hypotheses.len()
    }

    pub fn hypotheses(&self) -> &[Hypothesis] {
        self.hypotheses.as_slice()
    }

    pub fn with_hypotheses(hypotheses: Vec<Hypothesis>) -> Self {
        Self { hypotheses: log_normalize(hypotheses) }
    }

    pub fn all_tracks(&self) -> BTreeSet<usize> {
        self.hypotheses
            .iter()
            .flat_map(|h| h.tracks.iter())
            .copied()
            .collect()
    }

    pub fn track_as_indices(&self) -> Vec<usize> {
        self.all_tracks().iter().map(|t| t - 1).collect()
    }

    pub fn reindex(&mut self) {
        let mut old2new: BTreeMap<usize, usize> = self
            .hypotheses
            .iter()
            .flat_map(|h| h.tracks.iter())
            .map(|&t| (t, 0))
            .collect();
        for (k, v) in old2new.values_mut().enumerate() {
            *v = k + 1;
        }

        for hyp in &mut self.hypotheses {
            hyp.reindex(&old2new);
        }
    }

    pub fn into_reindexed(mut self) -> Self {
        self.reindex();
        self
    }
}
