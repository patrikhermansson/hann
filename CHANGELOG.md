## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/).

### [1.1.0] - 2025-03-10

#### Added

- Added initial implementation of `SSGIndex`.
- Introduced custom distance functions for user-defined similarity metrics.

#### Changed

- Optimized `HNSWIndex` neighbor selection algorithm.

#### Fixed

- Fixed an issue with graph serialization causing corrupted saves.

---

### [1.0.0] - 2025-03-01

#### Added

- Initial release with `HNSWIndex`, `BruteForceIndex`, and `RPTIndex`.
- Implemented basic indexing and querying functionality.
- Added benchmarks and test cases.
