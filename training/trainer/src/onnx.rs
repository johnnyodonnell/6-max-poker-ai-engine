//! Minimal ONNX writer: just enough protobuf to emit a ModelProto with an
//! inference graph and embedded float/int64 initializers. No protoc or code
//! generation — field numbers follow onnx/onnx.proto.
//!
//! Usage: build a `Graph` with `input`/`output`/`init_*`/`node`, then
//! `model_bytes(opset)`. Each `node` call returns its (auto-named) output.

// ---- protobuf wire encoding -------------------------------------------------

fn varint(buf: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

fn key(buf: &mut Vec<u8>, field: u32, wire: u32) {
    varint(buf, ((field << 3) | wire) as u64);
}

fn put_int(buf: &mut Vec<u8>, field: u32, v: i64) {
    key(buf, field, 0);
    varint(buf, v as u64);
}

fn put_bytes(buf: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    key(buf, field, 2);
    varint(buf, bytes.len() as u64);
    buf.extend_from_slice(bytes);
}

fn put_str(buf: &mut Vec<u8>, field: u32, s: &str) {
    put_bytes(buf, field, s.as_bytes());
}

fn put_f32(buf: &mut Vec<u8>, field: u32, v: f32) {
    key(buf, field, 5);
    buf.extend_from_slice(&v.to_le_bytes());
}

// ---- ONNX messages ------------------------------------------------------------

pub const FLOAT: i64 = 1;
pub const INT64: i64 = 7;

pub enum Attr {
    Int(&'static str, i64),
    Ints(&'static str, Vec<i64>),
    Float(&'static str, f32),
}

impl Attr {
    fn encode(&self) -> Vec<u8> {
        let mut b = Vec::new();
        match self {
            Attr::Int(name, v) => {
                put_str(&mut b, 1, name);
                put_int(&mut b, 3, *v);
                put_int(&mut b, 20, 2); // AttributeType.INT
            }
            Attr::Ints(name, vs) => {
                put_str(&mut b, 1, name);
                for &v in vs {
                    put_int(&mut b, 8, v);
                }
                put_int(&mut b, 20, 7); // INTS
            }
            Attr::Float(name, v) => {
                put_str(&mut b, 1, name);
                put_f32(&mut b, 2, *v);
                put_int(&mut b, 20, 1); // FLOAT
            }
        }
        b
    }
}

/// A tensor dimension in a graph input/output signature.
pub enum Dim {
    Fixed(i64),
    Named(&'static str),
}

fn value_info(name: &str, elem: i64, dims: &[Dim]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let mut dim = Vec::new();
        match d {
            Dim::Fixed(v) => put_int(&mut dim, 1, *v),
            Dim::Named(s) => put_str(&mut dim, 2, s),
        }
        put_bytes(&mut shape, 1, &dim);
    }
    let mut tensor = Vec::new();
    put_int(&mut tensor, 1, elem);
    put_bytes(&mut tensor, 2, &shape);
    let mut ty = Vec::new();
    put_bytes(&mut ty, 1, &tensor);
    let mut vi = Vec::new();
    put_str(&mut vi, 1, name);
    put_bytes(&mut vi, 2, &ty);
    vi
}

#[derive(Default)]
pub struct Graph {
    nodes: Vec<Vec<u8>>,
    inits: Vec<Vec<u8>>,
    inputs: Vec<Vec<u8>>,
    outputs: Vec<Vec<u8>>,
    counter: usize,
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        format!("{hint}_{}", self.counter)
    }

    pub fn input(&mut self, name: &str, elem: i64, dims: &[Dim]) -> String {
        self.inputs.push(value_info(name, elem, dims));
        name.to_string()
    }

    pub fn output(&mut self, name: &str, elem: i64, dims: &[Dim]) {
        self.outputs.push(value_info(name, elem, dims));
    }

    fn init_raw(&mut self, name: &str, dims: &[i64], elem: i64, raw: Vec<u8>) -> String {
        let mut t = Vec::new();
        for &d in dims {
            put_int(&mut t, 1, d);
        }
        put_int(&mut t, 2, elem);
        put_str(&mut t, 8, name);
        put_bytes(&mut t, 9, &raw);
        self.inits.push(t);
        name.to_string()
    }

    /// A float32 initializer (weights / constants), named uniquely from `hint`.
    pub fn init_f32(&mut self, hint: &str, dims: &[i64], data: &[f32]) -> String {
        assert_eq!(dims.iter().product::<i64>() as usize, data.len(), "{hint}: dims vs data");
        let name = self.fresh(hint);
        let raw = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.init_raw(&name, dims, FLOAT, raw)
    }

    pub fn init_i64(&mut self, hint: &str, dims: &[i64], data: &[i64]) -> String {
        let name = self.fresh(hint);
        let raw = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.init_raw(&name, dims, INT64, raw)
    }

    pub fn scalar_f32(&mut self, v: f32) -> String {
        self.init_f32("c", &[], &[v])
    }

    /// Add a node; returns its single output's name.
    pub fn node(&mut self, op: &str, inputs: &[&str], attrs: &[Attr]) -> String {
        let out = self.fresh(op);
        self.node_named(op, inputs, attrs, &out);
        out
    }

    /// Add a node whose output gets a fixed name (e.g. a graph output).
    pub fn node_named(&mut self, op: &str, inputs: &[&str], attrs: &[Attr], out: &str) {
        let mut n = Vec::new();
        for i in inputs {
            put_str(&mut n, 1, i);
        }
        put_str(&mut n, 2, out);
        let name = self.fresh("n");
        put_str(&mut n, 3, &name);
        put_str(&mut n, 4, op);
        for a in attrs {
            put_bytes(&mut n, 5, &a.encode());
        }
        self.nodes.push(n);
    }

    /// Serialize as a ModelProto (IR version 8, default-domain `opset`).
    pub fn model_bytes(&self, opset: i64) -> Vec<u8> {
        let mut g = Vec::new();
        for n in &self.nodes {
            put_bytes(&mut g, 1, n);
        }
        put_str(&mut g, 2, "poker_net");
        for t in &self.inits {
            put_bytes(&mut g, 5, t);
        }
        for v in &self.inputs {
            put_bytes(&mut g, 11, v);
        }
        for v in &self.outputs {
            put_bytes(&mut g, 12, v);
        }
        let mut op = Vec::new();
        put_str(&mut op, 1, "");
        put_int(&mut op, 2, opset);
        let mut m = Vec::new();
        put_int(&mut m, 1, 8); // ir_version
        put_str(&mut m, 2, "6-max-poker-ai-engine");
        put_bytes(&mut m, 7, &g);
        put_bytes(&mut m, 8, &op);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varints() {
        let mut b = Vec::new();
        varint(&mut b, 300);
        assert_eq!(b, [0xac, 0x02]);
        let mut b = Vec::new();
        put_int(&mut b, 1, -1);
        assert_eq!(b.len(), 11); // negative int64 -> 10-byte varint
    }
}
