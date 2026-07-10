use crate::hypothesis as hyp;
use ndarray::{self as nd, s, Array2, ArrayBase, ArrayView2, Data, Ix2};
use std::collections::{BTreeSet, HashMap, HashSet};

fn unique_with_counts(data: &[u8]) -> HashMap<u8, usize> {
    let mut counts: HashMap<u8, usize> = HashMap::new();
    for &x in data {
        *counts.entry(x).or_insert(0) += 1;
    }
    counts
}

fn find_merging_clusters(assoc_local: &ArrayView2<u8>) -> Vec<HashSet<usize>> {
    let cluster_idxs_might_merge = &assoc_local.row(0) - 1;
    let master_clusters_with_counts =
        unique_with_counts(cluster_idxs_might_merge.as_slice().unwrap());
    let merging_masters: Vec<_> = master_clusters_with_counts
        .iter()
        .filter_map(|(&c_idx, &count)| if count > 1 { Some(c_idx) } else { None })
        .collect();
    let merging_clusters: Vec<HashSet<_>> = merging_masters
        .iter()
        .map(|&merging_master| {
            cluster_idxs_might_merge
                .iter()
                .enumerate()
                .filter_map(|(idx, &cidx)| {
                    if cidx == merging_master {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .collect();

    merging_clusters
}

fn get_tracks_per_merging_cluster(
    merging_clusters: &[HashSet<usize>],
    prior_hypothses_per_cluster: &[hyp::Hypotheses],
) -> HashMap<usize, BTreeSet<usize>> {
    let relevant_clusters: HashSet<usize> = merging_clusters.iter().flatten().copied().collect();

    let tracks_per_merging_cluster = relevant_clusters
        .into_iter()
        .map(|cluster| {
            (
                cluster,
                prior_hypothses_per_cluster
                    .get(cluster)
                    .unwrap()
                    .all_tracks(),
            )
        })
        .collect();

    tracks_per_merging_cluster
}

fn find_linking_measurements(
    llr: &ArrayView2<f64>,
    merging_clusters: &[HashSet<usize>],
    tracks_per_merging_cluster: &HashMap<usize, BTreeSet<usize>>,
) -> HashMap<usize, HashSet<usize>> {
    let cols = llr.dim().1;
    let num_measurements = cols - 1;
    let mut m2c_map: HashMap<usize, HashSet<usize>> = HashMap::new();

    for cluster_idxs in merging_clusters {
        // This maps an enumeration index back to an actual cluster index
        let c_map: Vec<usize> = cluster_idxs.iter().copied().collect();

        // gated[c][j] == true iff any track of cluster c has a finite llr
        // in measurement column j+1. [num_clusters x num_measurements]
        let gated: Vec<Vec<bool>> = c_map
            .iter()
            .map(|cluster| {
                let tracks = &tracks_per_merging_cluster[cluster];
                (1..cols)
                    .map(|col| tracks.iter().any(|&t| llr[(t - 1, col)].is_finite()))
                    .collect()
            })
            .collect();

        for j in 0..num_measurements {
            // All clusters (by original index) that gated measurement j
            let gating_clusters: HashSet<usize> = c_map
                .iter()
                .zip(&gated)
                .filter(|(_, cluster_row)| cluster_row[j])
                .map(|(&cluster, _)| cluster)
                .collect();

            // Only measurements gated by more than one cluster are "linking"
            if gating_clusters.len() > 1 {
                debug_assert!(
                    !m2c_map.contains_key(&(j + 1)),
                    "Measurement {} already accounted for, should not be possible(?)",
                    j + 1
                );
                m2c_map.insert(j + 1, gating_clusters);
            }
        }
    }

    m2c_map
}

fn invert_lm2c_map(lm2c_map: &HashMap<usize, HashSet<usize>>) -> HashMap<usize, HashSet<usize>> {
    let mut c2m_map: HashMap<usize, HashSet<usize>> = HashMap::new();

    for (&meas, cluster_set) in lm2c_map {
        for &cluster in cluster_set {
            c2m_map.entry(cluster).or_default().insert(meas);
        }
    }

    c2m_map
}

#[derive(Debug, Clone)]
pub struct LinkingMappings {
    linking_measurement_to_clusters: HashMap<usize, HashSet<usize>>,
    cluster_to_linking_measurements: HashMap<usize, HashSet<usize>>,
}

impl LinkingMappings {
    pub fn linking_measurement_to_clusters(&self) -> &HashMap<usize, HashSet<usize>> {
        &self.linking_measurement_to_clusters
    }

    pub fn cluster_to_linking_measurements(&self) -> &HashMap<usize, HashSet<usize>> {
        &self.cluster_to_linking_measurements
    }

    pub fn all_cluster_idxs(&self) -> Vec<usize> {
        self.cluster_to_linking_measurements.keys().copied().collect()
    }

    pub fn all_linking_measurement_idxs(&self) -> Vec<usize> {
        self.linking_measurement_to_clusters.keys().copied().collect()
    }
    // pub fn new(
    //     linking_measurement_to_clusters: HashMap<usize, HashSet<usize>>,
    //     cluster_to_linking_measurements: HashMap<usize, HashSet<usize>>,
    // ) -> Self {
    //     Self {
    //         linking_measurement_to_clusters,
    //         cluster_to_linking_measurements,
    //     }
    // }
}

#[derive(Debug, Clone)]
pub struct ClusterLinks {
    merging_clusters: Vec<HashSet<usize>>,
    unmerging_clusters: HashSet<usize>,
    clusters_that_merge: HashSet<usize>,
    linking_mappings: LinkingMappings,
    // linking_measurement_to_clusters: HashMap<usize, HashSet<usize>>,
    // cluster_to_linking_measurements: HashMap<usize, HashSet<usize>>,
}

impl ClusterLinks {
    pub fn from_parsed_mat_file(
        llr: &ArrayView2<'_, f64>,
        prior_hypotheses_per_cluster: &[hyp::Hypotheses],
        assoc_local: &Array2<u8>,
    ) -> Self {
        let merging_clusters = find_merging_clusters(&assoc_local.view());
        let tracks_per_merging_cluster = get_tracks_per_merging_cluster(
            merging_clusters.as_slice(),
            prior_hypotheses_per_cluster,
        );
        let linking_measurement_to_clusters = find_linking_measurements(
            &llr,
            merging_clusters.as_slice(),
            &tracks_per_merging_cluster,
        );
        let cluster_to_linking_measurements = invert_lm2c_map(&linking_measurement_to_clusters);

        let clusters_that_merge: HashSet<_> = merging_clusters.iter().flatten().copied().collect();
        let unmerging_clusters = (0..assoc_local.dim().1)
            .filter(|cluster_idx| !clusters_that_merge.contains(cluster_idx))
            .collect();

        Self {
            merging_clusters,
            unmerging_clusters,
            clusters_that_merge,
            linking_mappings: LinkingMappings {
                linking_measurement_to_clusters,
                cluster_to_linking_measurements,
            },
        }
    }

    pub fn merging_clusters(&self) -> &Vec<HashSet<usize>> {
        &self.merging_clusters
    }

    pub fn unmerging_clusters(&self) -> &HashSet<usize> {
        &self.unmerging_clusters
    }

    pub fn clusters_that_merge(&self) -> &HashSet<usize> {
        &self.clusters_that_merge
    }

    // pub fn linking_measurement_to_clusters(&self) -> &HashMap<usize, HashSet<usize>> {
    //     &self.linking_measurement_to_clusters
    // }

    // pub fn cluster_to_linking_measurements(&self) -> &HashMap<usize, HashSet<usize>> {
    //     &self.cluster_to_linking_measurements
    // }

    pub fn linking_mappings(&self) -> &LinkingMappings {
        &self.linking_mappings
        // LinkingMappings {
        //     linking_measurement_to_clusters: self.linking_measurement_to_clusters.clone(),
        //     cluster_to_linking_measurements: self.cluster_to_linking_measurements.clone(),
        // }
    }
}

fn edmund_to_lc<D: Data<Elem = f64>>(llr_edmund: &ArrayBase<D, Ix2>) -> Array2<f64> {
    let (n, mpn) = llr_edmund.dim();
    let m = mpn - n;
    let mp1 = m + 1;
    let misdetection_block = llr_edmund.slice(s![.., m..]);
    let right_diag = misdetection_block.diag();
    let mut llr_lc = Array2::zeros((n, mp1));
    llr_lc.column_mut(0).assign(&right_diag);
    llr_lc
        .slice_mut(s![.., 1..])
        .assign(&llr_edmund.slice(s![.., ..m]));

    llr_lc
}

#[cfg(test)]
mod tests {
    use core::f64;
    use std::vec;

    use super::*;

    #[test]
    fn test_measurement_to_cluster_map() {
        //         assocLocal = np.array([
        //     [1, 1],
        //     [1, 0]
        // ])
        let assoc_local = nd::arr2(&[
            [1, 1], //
            [1, 0],
        ]);

        let inf = f64::INFINITY;
        let llr_edmund = nd::arr2(&[
            [3.0, -inf, -0.60, -inf, -inf, -inf, -inf],
            [3.2, -inf, -inf, -0.56, -inf, -inf, -inf],
            [-3.0, 1.2, -inf, -inf, -0.46, -inf, -inf],
            [-inf, 3.0, -inf, -inf, -inf, -0.62, -inf],
            [-inf, -0.4, -inf, -inf, -inf, -inf, -0.55],
        ]);
        let llr = edmund_to_lc(&llr_edmund.view());

        let prior_hypotheses_per_cluster = vec![
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![1, 2], 0.5f64.ln()),
                hyp::Hypothesis::new(vec![1, 3], 0.5f64.ln()),
            ]),
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![4], 0.5f64.ln()),
                hyp::Hypothesis::new(vec![5], 0.5f64.ln()),
            ]),
        ];

        let cluster_links = ClusterLinks::from_parsed_mat_file(
            &llr.view(),
            prior_hypotheses_per_cluster.as_slice(),
            &assoc_local,
        );

        let expected_mapping: HashMap<_, _> = HashMap::from([(2, HashSet::from([0, 1]))]);

        assert_eq!(
            &expected_mapping,
            cluster_links.linking_mappings().linking_measurement_to_clusters()
        );
    }

    #[test]
    fn test_measurement_to_cluster_map2() {
        //         assocLocal = np.array([
        //     [1, 1],
        //     [1, 0]
        // ])
        let assoc_local = nd::arr2(&[
            [1, 1, 1], //
            [1, 0, 0],
        ]);

        let inf = f64::INFINITY;
        let llr = nd::arr2(&[
            [0.1, 1.0, -inf],
            [0.1, 1.0, -inf],
            [0.1, 1.0, 1.0],
            [0.1, -inf, 1.0],
            [0.1, -inf, 1.0],
            [0.1, 1.0, -inf],
        ]);

        let prior_hypotheses_per_cluster = vec![
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![1], 0.5f64.ln()),
                hyp::Hypothesis::new(vec![2], 0.5f64.ln()),
            ]),
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![3], 0.5f64.ln()),
                hyp::Hypothesis::new(vec![4], 0.5f64.ln()),
            ]),
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![5], 0.5f64.ln()),
                hyp::Hypothesis::new(vec![6], 0.5f64.ln()),
            ]),
        ];

        let cluster_links = ClusterLinks::from_parsed_mat_file(
            &llr.view(),
            prior_hypotheses_per_cluster.as_slice(),
            &assoc_local,
        );

        let expected_mapping: HashMap<_, _> =
            HashMap::from([(1, HashSet::from([0, 1, 2])), (2, HashSet::from([1, 2]))]);

        assert_eq!(
            &expected_mapping,
            cluster_links.linking_mappings().linking_measurement_to_clusters()
        );
    }

    #[test]
    fn test_measurement_to_cluster_map3() {
        //         assocLocal = np.array([
        //     [1, 1],
        //     [1, 0]
        // ])
        let assoc_local = nd::arr2(&[
            [1, 1, 3, 3, 1, 6], //
            [1, 0, 1, 0, 0, 1],
        ]);

        let inf = f64::INFINITY;
        let llr = nd::arr2(&[
            [0.1, 1.0, -inf, -inf, -inf, -inf, -inf],
            [0.1, 1.0, -inf, -inf, -inf, -inf, -inf],
            [0.1, 1.0, 1.0, -inf, -inf, -inf, -inf],
            [0.1, -inf, 1.0, -inf, -inf, -inf, -inf],
            [0.1, -inf, -inf, 1.0, -inf, -inf, -inf],
            [0.1, -inf, -inf, 1.0, 1.0, -inf, -inf],
            [0.1, -inf, -inf, -inf, 1.0, -inf, -inf],
            [0.1, -inf, -inf, 1.0, 1.0, 1.0, -inf],
            [0.1, -inf, -inf, -inf, -inf, 1.0, -inf],
            [0.1, -inf, -inf, -inf, -inf, -inf, -inf],
            [0.1, 1.0, -inf, -inf, -inf, -inf, -inf],
            [0.1, -inf, -inf, -inf, -inf, -inf, 1.0],
        ]);

        let prior_hypotheses_per_cluster = vec![
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![1], 0.5f64.ln()),
                hyp::Hypothesis::new(vec![2], 0.5f64.ln()),
            ]),
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![3], 0.5f64.ln()),
                hyp::Hypothesis::new(vec![4], 0.5f64.ln()),
            ]),
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![5], 0.3f64.ln()),
                hyp::Hypothesis::new(vec![6], 0.3f64.ln()),
                hyp::Hypothesis::new(vec![7], 0.4f64.ln()),
            ]),
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![8], 0.3f64.ln()),
                hyp::Hypothesis::new(vec![9], 0.3f64.ln()),
                hyp::Hypothesis::new(vec![10], 0.4f64.ln()),
            ]),
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![11], 0.9f64.ln()),
                hyp::Hypothesis::new(vec![], 0.1f64.ln()),
            ]),
            hyp::Hypotheses::with_hypotheses(vec![
                hyp::Hypothesis::new(vec![12], 0.9f64.ln()),
                hyp::Hypothesis::new(vec![], 0.1f64.ln()),
            ]),
        ];

        let cluster_links = ClusterLinks::from_parsed_mat_file(
            &llr.view(),
            prior_hypotheses_per_cluster.as_slice(),
            &assoc_local,
        );

        let expected_mapping: HashMap<_, _> = HashMap::from([
            (1, HashSet::from([0, 1, 4])),
            (3, HashSet::from([2, 3])),
            (4, HashSet::from([2, 3])),
        ]);

        assert_eq!(
            &expected_mapping,
            cluster_links.linking_measurement_to_clusters()
        );
    }
}
