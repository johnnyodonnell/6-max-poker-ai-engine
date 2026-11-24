//! Shared helpers for the parity tests: load fixtures dumped by
//! `cfr/scripts/dump_fixtures.ts`.

use std::path::PathBuf;

use serde_json::Value;

pub fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../fixtures").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "read {}: {e}\nGenerate fixtures locally with `npx tsx cfr/scripts/dump_fixtures.ts` \
             and sync them with cfr/sync_to_box.sh",
            path.display()
        )
    });
    serde_json::from_str(&text).expect("fixture json")
}
