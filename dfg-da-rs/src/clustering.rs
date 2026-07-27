//! Decode PMBM cluster membership from the CSR / "jagged" cloud arrays stored
//! in `priorLikelihood*.mat`.
//!
//! Clusters are stored as a two-level nesting `cluster -> hypotheses -> tracks`,
//! each level a flat `values` buffer grouped by a per-group `card`inality array:
//!
//! ```text
//! clusters / clusters_card : cluster    -> hypothesis ids (1-based)
//! hypos    / hypos_card     : hypothesis -> track numbers  (1-based)
//! ```
//!
//! There are no explicit boundaries; group `i` owns `values[off[i]..off[i+1]]`
//! with `off = prefix_sum(card)`. [`Jagged`] is that generic primitive;
//! [`Clustering`] composes two of them to answer cluster<->track queries.

use std::collections::BTreeSet;
use std::fmt;

/// Generic CSR / jagged array: a `values` buffer plus `offsets` (length
/// `n_groups + 1`) that delimit the groups. Group `i` is
/// `values[offsets[i]..offsets[i + 1]]`.
#[derive(Clone, Debug)]
pub struct Jagged<T> {
    offsets: Vec<usize>,
    values: Vec<T>,
}

impl<T> Jagged<T> {
    /// Build from a flat `values` buffer and per-group cardinalities.
    ///
    /// Enforces the structural contract `sum(card) == values.len()`; returns
    /// [`ClusterError::CardMismatch`] otherwise.
    pub fn from_card(values: Vec<T>, card: &[usize]) -> Result<Self, ClusterError> {
        let mut offsets = Vec::with_capacity(card.len() + 1);
        offsets.push(0);
        for &c in card {
            offsets.push(offsets.last().unwrap() + c);
        }
        let total = *offsets.last().unwrap();
        if total != values.len() {
            return Err(ClusterError::CardMismatch {
                sum_card: total,
                n_values: values.len(),
            });
        }
        Ok(Self { offsets, values })
    }

    /// Number of groups.
    #[inline]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The items of group `i`. Panics if `i >= len()`.
    #[inline]
    pub fn row(&self, i: usize) -> &[T] {
        &self.values[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Iterate over every group as a slice.
    pub fn iter(&self) -> impl Iterator<Item = &[T]> + '_ {
        (0..self.len()).map(move |i| self.row(i))
    }
}

/// Errors from decoding the cloud arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClusterError {
    /// A `(values, card)` pair is inconsistent.
    CardMismatch { sum_card: usize, n_values: usize },
    /// A hypothesis id in `clusters` is 0 or exceeds the number of hypotheses.
    HypothesisIdOutOfRange { id: usize, n_hypotheses: usize },
}

impl fmt::Display for ClusterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ClusterError::CardMismatch { sum_card, n_values } => write!(
                f,
                "malformed jagged array: sum(card)={sum_card} != len(values)={n_values}"
            ),
            ClusterError::HypothesisIdOutOfRange { id, n_hypotheses } => {
                write!(f, "hypothesis id {id} out of range 1..={n_hypotheses}")
            }
        }
    }
}

impl std::error::Error for ClusterError {}

/// Decoded cluster membership for one scan.
///
/// Holds `cluster -> sorted track numbers` (as a [`Jagged`]) and the inverse
/// `track -> cluster` lookup. Track numbers and cluster ids follow the file's
/// 1-based track numbering, but cluster **indices** here are 0-based.
pub struct Clustering {
    cluster_tracks: Jagged<usize>,
    /// `track_of[t]` = 0-based cluster index of track `t`, or `usize::MAX` if `t`
    /// is not a real track number (index 0 is always unused: tracks are 1-based).
    track_of: Vec<usize>,
}

impl Clustering {
    /// Decode from the four raw cloud arrays (as stored in the `.mat`, 1-based).
    ///
    /// Zero-copy friendly: pass the borrowed slices you get from
    /// `PyReadonlyArray1::as_slice()`.
    pub fn decode(
        clusters: &[usize],
        clusters_card: &[usize],
        hypos: &[usize],
        hypos_card: &[usize],
    ) -> Result<Self, ClusterError> {
        let hypos_j = Jagged::from_card(hypos.to_vec(), hypos_card)?;
        let clusters_j = Jagged::from_card(clusters.to_vec(), clusters_card)?;

        // Compose: for each cluster, union the track sets of its hypotheses.
        let mut flat = Vec::new();
        let mut card = Vec::with_capacity(clusters_j.len());
        let mut max_track = 0usize;
        for c in 0..clusters_j.len() {
            let mut set = BTreeSet::new(); // sorted + deduped
            for &hid in clusters_j.row(c) {
                if hid == 0 || hid > hypos_j.len() {
                    return Err(ClusterError::HypothesisIdOutOfRange {
                        id: hid,
                        n_hypotheses: hypos_j.len(),
                    });
                }
                for &t in hypos_j.row(hid - 1) {
                    set.insert(t);
                    max_track = max_track.max(t);
                }
            }
            card.push(set.len());
            flat.extend(set);
        }
        let cluster_tracks = Jagged::from_card(flat, &card)?;

        // Inverse lookup, indexed directly by (1-based) track number.
        let mut track_of = vec![usize::MAX; max_track + 1];
        for c in 0..cluster_tracks.len() {
            for &t in cluster_tracks.row(c) {
                track_of[t] = c;
            }
        }

        Ok(Self {
            cluster_tracks,
            track_of,
        })
    }

    /// Number of clusters.
    #[inline]
    pub fn n_clusters(&self) -> usize {
        self.cluster_tracks.len()
    }

    /// Number of distinct tracks.
    #[inline]
    pub fn n_tracks(&self) -> usize {
        self.cluster_tracks.iter().map(<[usize]>::len).sum()
    }

    /// Sorted track numbers in cluster `c` (0-based index). Panics if out of range.
    #[inline]
    pub fn tracks_in(&self, c: usize) -> &[usize] {
        self.cluster_tracks.row(c)
    }

    /// 0-based cluster index of `track` (1-based track number), or `None`.
    #[inline]
    pub fn cluster_of(&self, track: usize) -> Option<usize> {
        self.track_of
            .get(track)
            .copied()
            .filter(|&c| c != usize::MAX)
    }

    /// Iterate `(cluster_index, &[track numbers])`.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &[usize])> + '_ {
        self.cluster_tracks.iter().enumerate()
    }

    /// Materialize as `Vec<Vec<usize>>` (handy for returning across the pyo3 boundary).
    pub fn to_vecs(&self) -> Vec<Vec<usize>> {
        self.cluster_tracks.iter().map(<[usize]>::to_vec).collect()
    }
}

/// Group clusters into superclusters from `assocLocal` row 0 (the 1-based master
/// id per cluster). Returns, per supercluster, the 0-based cluster indices in it.
pub fn superclusters(assoc_master_ids: &[usize]) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (c, &master) in assoc_master_ids.iter().enumerate() {
        groups.entry(master).or_default().push(c);
    }
    groups.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_two_clusters() {
        // cluster 0 owns hypotheses {1,2}; cluster 1 owns {3}.
        // hypo1 -> [1,2], hypo2 -> [1,3], hypo3 -> [4]
        let c = Clustering::decode(&[1, 2, 3], &[2, 1], &[1, 2, 1, 3, 4], &[2, 2, 1]).unwrap();
        assert_eq!(c.n_clusters(), 2);
        assert_eq!(c.n_tracks(), 4);
        assert_eq!(c.tracks_in(0), &[1, 2, 3]);
        assert_eq!(c.tracks_in(1), &[4]);
        assert_eq!(c.cluster_of(3), Some(0));
        assert_eq!(c.cluster_of(4), Some(1));
        assert_eq!(c.cluster_of(99), None);
    }

    #[test]
    fn detects_malformed() {
        assert!(matches!(
            Clustering::decode(&[1, 2], &[3], &[1], &[1]),
            Err(ClusterError::CardMismatch { .. })
        ));
    }

    #[test]
    fn empty_scan() {
        let c = Clustering::decode(&[], &[], &[], &[]).unwrap();
        assert_eq!(c.n_clusters(), 0);
        assert_eq!(c.n_tracks(), 0);
    }

    #[test]
    fn supercluster_grouping() {
        // clusters 0 and 2 share master 1; cluster 1 has master 2.
        assert_eq!(superclusters(&[1, 2, 1]), vec![vec![0, 2], vec![1]]);
    }
}
