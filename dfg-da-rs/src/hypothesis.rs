pub struct Hypothesis {
    log_prob: f64,
    tracks: Vec<usize>
}

pub struct Hypotheseses {
    hypotheseses: Vec<Hypothesis>
}
