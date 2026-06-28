use super::{ComputeGraph, ComputeNode};
use crate::context::WgpuContext;
use std::sync::Arc;

pub fn fuse_compute_graphs(ctx: Arc<WgpuContext>, graphs: &[&ComputeGraph]) -> ComputeGraph {
    let mut fused = ComputeGraph::new(ctx);

    for graph in graphs {
        for node in &graph.nodes {
            let cloned = match node {
                ComputeNode::Wgpu {
                    name,
                    pipeline,
                    bind_group,
                    workgroups,
                } => ComputeNode::Wgpu {
                    name: name.clone(),
                    pipeline: pipeline.clone(),
                    bind_group: bind_group.clone(),
                    workgroups: *workgroups,
                },
                #[cfg(feature = "cuda")]
                ComputeNode::Cuda {
                    name,
                    bindings,
                    workgroups,
                } => ComputeNode::Cuda {
                    name: name.clone(),
                    bindings: bindings
                        .iter()
                        .map(|b| super::CudaNodeBinding {
                            binding: b.binding,
                            slice: b.slice.clone(),
                            mode: b.mode,
                            size: b.size,
                            cached_meta: b.cached_meta.clone(),
                        })
                        .collect(),
                    workgroups: *workgroups,
                },
            };
            fused.nodes.push(cloned);
        }
    }

    fused
}
