## Summary

<!-- What changed and why. -->

## Release candidate

When this PR is opened/updated, CI builds all platforms and publishes a **Release Candidate**
prerelease tagged `vX.Y.Z-rc` on the [Releases page](../../releases). Download the binary for
each platform and test it **on real hardware** before ticking the boxes below.

> PRs into `master` are only accepted from the `dev` branch. PRs from any other branch are
> rejected automatically by the `guard-source-branch` check.

## Manual cross-platform test

Run the full manual suite on each platform — `tests/tests.sh` (unix) or `tests/tests.ps1`
(Windows) — and confirm the version is correct. Tick a box only after the binary passes on
**real hardware** for that platform. All boxes must be checked before this PR can be merged.

<!-- TEST-CHECKLIST-START -->
- [ ] linux-x86_64 — tested on real hardware, `tests/tests.sh` ✓, version correct
- [ ] linux-aarch64 — tested on real hardware, `tests/tests.sh` ✓, version correct
- [ ] windows-x86_64 — tested on real hardware, `tests/tests.ps1` ✓, version correct
- [ ] windows-aarch64 — tested on real hardware, `tests/tests.ps1` ✓, version correct
- [ ] darwin-x86_64 — tested on real hardware, `tests/tests.sh` ✓, version correct
- [ ] darwin-arm64 — tested on real hardware, `tests/tests.sh` ✓, version correct
- [ ] freebsd-x86_64 — tested on real hardware, `tests/tests.sh` ✓, version correct
- [ ] Hardware acceleration (NVENC / AMF / VPL) verified where applicable
<!-- TEST-CHECKLIST-END -->

## Notes

<!-- Anything reviewers should know. -->
