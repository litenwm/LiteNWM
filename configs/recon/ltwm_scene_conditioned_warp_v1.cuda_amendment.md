# LTWM scene-conditioned warp CUDA amendment

The frozen scene-warp run reached batch 1700 of epoch 1 before CUDA raised
`cudaErrorIllegalAddress`. No epoch completed and no checkpoint or validation
result was produced.

The retry changes only the numeric kernel boundary: the 27,530-parameter flow
MLP, `affine_grid`, and `grid_sample` execute in float32 outside bf16 autocast.
The architecture, inputs, loss weights, seed, optimizer, batch size, epoch
count, checkpoint rule, and validation gates remain those frozen in
`configs/recon/ltwm_scene_conditioned_warp_v1.json`.

This is an execution-safety amendment made before any validation result
existed. It is not a model or threshold selection.
