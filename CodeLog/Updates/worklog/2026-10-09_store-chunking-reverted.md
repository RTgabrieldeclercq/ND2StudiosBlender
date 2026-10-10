# Store: drop the chunked writes; the S: drive is failing, not limited

- **Date:** 2026-10-09
- **Author:** hyper
- **Branch:** ui-lag-troubleshooting
- **Base:** e1ae0e28

## What changed

- **Reverted.** `nodegraph/analysis_store.py` writes as it did before `dc55c6d`: one `write`
  for JSON, pyarrow's own file write for Parquet. `WRITE_CHUNK` and the in-memory Parquet
  buffer are removed.
- **Kept.** A write that fails mid-file now removes its `*.tmp` (`_discard_tmp`). Since
  `dc55c6d` a failed save no longer fails the run, so without the cleanup a broken store would
  collect half-written temp files.
- **Tests.** `test_analysis_store_writes` now checks:
  - a write that fails mid-file leaves no temp file, for JSON and Parquet;
  - a unit whose save raises keeps its result;
  - the output's note says so.
- **Docs.** The `dc55c6d` worklog is corrected to match.

## Why

`dc55c6d` chunked the store's writes because S: "accepts 16 MiB in one write and refuses
24 MiB". That was not a property of the drive: **S: is failing.** After the chunked version
went in:
- file CREATION on S: started failing with EINVAL for names of 28 characters or more, a limit
  NTFS does not have;
- concurrent saves all failed.

The System event log explains it. On 2026-10-09 it holds about 74,000 storage warnings since
10:54, when the log wrapped:
- 73,761 × `iaStorE` (Intel RST/VROC) event 4155, "I/O on 9RJ9KY1C has failed";
- 655 × NTFS event 140, "The system failed to flush data to the transaction log. Corruption
  may occur in VolumeId: S: … The I/O device reported an I/O error".

The CRC early-run ND2's "Windows fatal exception: page error" is the same failure, seen on
read.

So the chunking was a workaround for a symptom, carrying a false comment, and it is removed.
Not failing the run on a failed save is right for any store that cannot be written (failing,
full, read-only), so that part stays.

**For whoever uses S: next.** The volume needs its RAID or disk status checked, and its data
copied off, before more work is written to it.

## Files

- `nodegraph/analysis_store.py`: chunked writes removed; `_discard_tmp`.
- `nodegraph/selftest.py`: `test_analysis_store_writes` rewritten.
- `CodeLog/Updates/worklog/2026-10-09_field-playground-fixes.md`: fix 3 corrected.
- Regenerated: `codemap/gen/*`, `codemap/STATE.md`.

Shared files both developers edit: `nodegraph/selftest.py`, `codemap/gen/*`.

## How to verify

- `run_named.py . test_analysis_store_writes test_analysis_store`.
- In PowerShell: `Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='iaStorE'}
  -MaxEvents 5` shows the disk errors.

## Gates

- [x] `PYTHONUTF8=1 python -B -m nodegraph.selftest`: gated together with the playground
  commit; see `2026-10-09_field-playground-graph.md`.
- [x] `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png`: as above.
- [x] `python scripts/_catalog_snapshot.py` -> CATALOG IDENTICAL (no catalog change).
- [x] `python scripts/_codemap.py` -> CODEMAP CURRENT
- [ ] `python scripts/_sync_check.py` -> IN SYNC: not pushed.
