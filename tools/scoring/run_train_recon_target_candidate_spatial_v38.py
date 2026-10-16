#!/usr/bin/env python3
"""Run v37 candidate ranker with real 258-D spatial LTWM rollout features."""
from __future__ import annotations
from pathlib import Path

def main():
 target=Path(__file__).with_name("train_recon_target_candidate_rank_v37.py");source=target.read_text()
 old='x,valid=compact_base_feature(feature,score,np.empty((100,0),np.float32))'
 new='x,valid=compact_base_feature(feature,score,np.empty((100,0),np.float32))\n with np.load(source.parent/"spatial_rollout_v17.npz",allow_pickle=False) as rp: x=np.concatenate((x,rp["spatial_rollout_features"][:,1:].astype(np.float32)),axis=2)'
 variants='(("candidate_universal_support",slice(474,486)),("candidate_motion_universal_support",slice(434,486)))'
 spatial='(("candidate_universal_support",slice(474,486)),("candidate_motion_universal_support_spatial",slice(434,744)))'
 source=source.replace(old,new).replace(variants,spatial).replace('"candidate_motion_universal_support"', '"candidate_motion_universal_support_spatial"').replace('recon_target_candidate_rank_v37_member','recon_target_candidate_spatial_v38_member').replace('recon_target_candidate_rank_v37','recon_target_candidate_spatial_v38')
 if old in source or variants in source:raise RuntimeError("v38 patch failed")
 exec(compile(source,str(target),"exec"),{"__name__":"__main__","__file__":str(target)})
if __name__=="__main__":main()
