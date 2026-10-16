# v70 external assets

Do not copy these machine-local directories into the v70 archive:

- `models/facebook/dinov3-vitl16-pretrain-lvd1689m/`
- `visualnav-transformer/`
- `diffusion_policy/`

The first is the upstream DINOv3 ViT-L/16 backbone. It is required when the
recipient needs to encode images, but it is not a project-trained v70 weight.

The second is the external NoMaD repository and normally contains the upstream
NoMaD checkpoint. The third supplies NoMaD's `ConditionalUnet1D`
implementation. They are needed only when the recipient must regenerate the
initial NoMaD candidate pool from images; they are unnecessary when frozen K17
candidate tensors are supplied.

Keep these as explicit external dependencies with local paths configured in
`configs/local_assets.json`. Project-owned v70 weights are packaged separately.
