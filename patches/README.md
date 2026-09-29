# Vendored source patches

## `zlib-1.3.2-cve-2026-85091.patch`

Applied to the upstream zlib **1.3.2** release tarball by the `zlib_builder`
stage in the `Dockerfile`.

Backports the single fix for a heap buffer overflow that is on zlib's `develop`
branch but in no tagged release (1.3.2 is the latest tag and is affected;
`master` is still at 1.3.2):

| CVE | Upstream commit | Fix |
| --- | --- | --- |
| CVE-2026-85091 | `df84af25` | Clear the stale `next_in`/`avail_in` pointers into the caller's buffer on the `gz_write` error path |

The overflow is in `gz_vacate()` and is reachable only through the `gzFile` API:
`gzprintf()`/`gzvprintf()` on a **non-blocking** `gzFile` after a write stall
leaves `strm.next_in` pointing into a buffer the caller has since released, and
the next `memmove()` runs past the end of the internal input buffer. Nothing in
this image opens a `gzFile` in non-blocking mode, so the practical exposure here
is low - but the fix is four lines, so it is applied rather than suppressed.

### How it was produced

Unmodified `git format-patch` output for upstream commit `df84af25`, which
touches only `gzwrite.c`. Unlike the tesseract patch there was nothing to
resolve: the hunk context matches 1.3.2 verbatim, and the `state->again` field
it relies on already exists in 1.3.2's `gzguts.h`. It was verified with
`git apply --check` against a pristine `v1.3.2` tarball.

### Note on scanner output

Patching does not change `ZLIB_VERSION`, which stays `"1.3.2"`. The scan gate
passes because the `zlib` apk-db record is removed in the runtime stage, the
same way libtiff is handled - not because a scanner can see the fix. The
`zlib.ZLIB_RUNTIME_VERSION` assertion in the `Dockerfile`'s gate block is there
to catch the opposite mistake: a drop-in that silently did not happen.

### When to remove this

Once Alpine packages a zlib containing `df84af25` - most likely whenever
upstream tags 1.3.3 or later - delete this patch and the `zlib_builder` stage,
and drop the `zlib` line from the `drop_apk_pkg.py` invocation.

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
