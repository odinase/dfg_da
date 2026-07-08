pub mod mcmh_solver;
pub mod per_cluster_mh;
pub mod mcmh_lbp;


pub struct MarginalSolver {
    solver: Box<dyn mcmh_solver::McMhSolver>,
}