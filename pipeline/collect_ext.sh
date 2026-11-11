#!/bin/bash
# Packs the extension results for the manuscript: ext_results.tgz in /work/dod220001/netrev
cd /work/dod220001/netrev
tar czf ext_results.tgz \
  $(ls out3_*/eval2_{ops,accuracy,accounting,dataset,matched_boot}* out3_*/eval3_a2_ops_* 2>/dev/null) \
  $(ls out2_*/eval3_a2_ops_* out2_*/eval2_matched_boot_*_ext_* 2>/dev/null) \
  $(ls logs/net3_eval_*.out 2>/dev/null)
ls -la ext_results.tgz; tar tzf ext_results.tgz | wc -l
