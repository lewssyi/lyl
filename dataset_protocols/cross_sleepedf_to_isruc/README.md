# SleepEDF -> ISRUC Cross-Dataset Protocol

This protocol is isolated from the original single-dataset protocols.

- NPZ view: `/home/b502/wssyi/LIME-for-Time-Series/tmp/cross_sleepedf_to_isruc_npz`
- Source train/val: `/home/b502/wssyi/SleepEDF/npz_epoch3`
- Target test: `/home/b502/wssyi/ISRUC-Sleep/npz_epoch3`
- Target files are used only as test data.

Run with:

```powershell
python tools\run_main_with_protocol_split.py --split-json /home/b502/wssyi/LIME-for-Time-Series/dataset_protocols/cross_sleepedf_to_isruc/split.json --dataset-preset edf_c0_no_concept --seed 42
```
