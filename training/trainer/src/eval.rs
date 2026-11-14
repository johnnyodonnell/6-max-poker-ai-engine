//! Snapshot evaluation (Phase 6).

use std::path::Path;

/// Called by the training daemon after each snapshot.
pub fn after_snapshot(_run_dir: &str, _snapshot: &Path, _args: &[String]) {}
