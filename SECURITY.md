# Security Policy

## Reporting a Vulnerability

`statumen` reads whole-slide image containers and passes compressed tile
payloads into codec backends. If you find a crash, memory-safety issue,
malformed-output bug, metadata leak, or unexpected file-system behavior, please
report it privately rather than opening a public issue.

Use GitHub's private vulnerability reporting for the repository, or contact the
maintainer through the repository owner profile if private reporting is not yet
enabled.

Please include:

- A minimal reproducer, including the smallest input file or generated fixture
  you can share.
- Rust version, target triple, operating system, and cargo features used.
- The API call, CLI command, or OpenSlide-shim call surface.
- Expected vs. observed behavior.

Reports are acknowledged within 7 days. Patches are issued as soon as possible,
generally within 30 days for high-severity issues.

## Supported Versions

The supported line is the latest published 0.2.x release and the current
`main` branch. Optional Metal behavior is triaged on supported macOS hardware
when that hardware is available.
