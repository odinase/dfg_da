#[derive(Debug, Clone)]
pub struct Hypothesis {
    log_weight: f64,
    tracks: Vec<usize>
}

#[derive(Debug, Clone)]
pub struct Hypotheses {
    hypotheseses: Vec<Hypothesis>
}
