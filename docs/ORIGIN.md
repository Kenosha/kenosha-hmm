# Origin and release status

Extracted from Gianluca Vagliano’s original 2022 Python implementation in
Kenosha/ATMAN-v1 at commit 872ef92054df00d72a6eefd5e10841427bfa0217.
The original author/date headers and algorithm references are retained. The
implementation was written from papers, algorithm descriptions, pseudocode and
text, without literal translation of third-party programming-language source.

The standalone library is released under the [MIT licence](../LICENSE).
Version 0.1.0 is the first standalone release, prepared on 28 October 2026.
The original implementation date and this packaging/release date are distinct.
ATMAN-specific postprocessing, data and model weights are outside this library.

The numerical routines retain the original algorithms. Packaging removes ATMAN
imports, project-root discovery and tennis postprocessing. The public convenience
API validates input and converts arrays into the original pandas representation.
See CHANGELOG.md for any numerical corrections required by independent tests.
