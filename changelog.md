# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes should be categorised under the following types:

- Added for new features.
- Changed for changes in existing functionality.
- Deprecated for soon-to-be removed features.
- Removed for now removed features.
- Fixed for any bug fixes.
- Security in case of vulnerabilities.

Where appropriate, links to specific Issues & Merge Requests are appreciated.

## [Unreleased]

### Added
- This changelog [!18](https://gitlab.com/emd-dev/emd/-/merge_requests/18)

### Fixed
- Input array dimensions in phase_align clarified and fixed up [ef28b36c](https://gitlab.com/emd-dev/emd/-/commit/ef28b36cac8be7224280fd7ba02d25b3f084ab30)

### Changed
- get_control_points internal refector [af153ed6](https://gitlab.com/emd-dev/emd/-/commit/af153ed606601f3963c125329c86710e47c06b45)

## [0.3.0] - 2020-07-22

### Added
- get_cycle_stat refectored to allow general numpy and user-specified metrics to be computed
- Logger coverage increased, particularly in cycle.py
  - Logger exit message added

### Changed
- Major SiftConfig refactor - API & syntax now much cleaner

## [0.2.0] - 2020-06-05

### Added
- Tutorials on the sift, hilbert-huang and holospectrum analyses.
- Parabolic extrema interpolation
- Average envelope scaling in sift
- Testing expanded to include python 3.5, 3.6, 3.7 & 3.8


### Changed
- API in sift functions updated for compatabillity with new SiftConfig
  - Expose options for extrema padding to top level sift function
  - Sift relevant util functions moved into sift.py submodule
- Masked sift functions merged into single function
- get_cycle_chain refactor to cleaner internal syntax


## [0.1.0]

### Added
- Everything