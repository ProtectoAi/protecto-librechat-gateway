# Vendored source patches

## `tesseract-5.5.3-cve-fixes.patch`

Applied to the upstream Tesseract **5.5.3** release tarball by the
`tesseract_builder` stage in the `Dockerfile`.

Backports three security fixes that are merged on Tesseract's `main` branch but
are not in any tagged release (5.5.3 is the latest; there is no 5.5.4):

| CVE | Upstream commit | Fix |
| --- | --- | --- |
| CVE-2026-88051 | `56e09ca1` | Validate vector counts in `GenericVector::read` |
| CVE-2026-88052 | `2d04d640` | Reject unicharset files whose inserts desync id from unichars |
| CVE-2026-88053 | `8b057468` | Fix out-of-bounds writes in `.traineddata` inttemp deserialization |

All three are out-of-bounds reads/writes reachable only by loading a crafted
`.traineddata` model file.

### How it was produced

The three commits do not cherry-pick cleanly onto 5.5.3: `56e09ca1` conflicts on
`Makefile.am` and `8b057468` conflicts on `src/classify/intproto.cpp`. This patch
is the resolved result of:

1. Applying each commit restricted to `src/` (`git apply -3 --include='src/*'`),
   which drops the `Makefile.am` / `unittest/` hunks — the unit tests are never
   built here.
2. Resolving the one remaining conflict, the `INT_TEMPLATES_STRUCT` constructor,
   in favour of upstream. That is the correct side: the same patch converts
   `Class` / `ClassPruners` from raw C arrays to value-initialized
   `std::array<...>{}` in `intproto.h`, so the in-class initialization the
   upstream constructor relies on is present.

### When to remove this

Once Alpine packages tesseract-ocr >= 5.5.4, delete this patch and the
`tesseract_builder` stage, and go back to `apk add tesseract-ocr
tesseract-ocr-data-eng` in the runtime stage.
