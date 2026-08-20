#!/usr/bin/env bash
# Downloads the open-access PDFs referenced in README.md into papers/pdfs/
# (gitignored -- re-run this instead of committing the PDFs).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p pdfs

fetch() {
  local name="$1" url="$2"
  if [ -f "pdfs/${name}.pdf" ]; then
    echo "skip (exists): ${name}.pdf"
    return
  fi
  echo "fetching ${name}.pdf ..."
  curl -sL "$url" -o "pdfs/${name}.pdf"
}

fetch ltc_hasani2021          "https://arxiv.org/pdf/2006.04439"
fetch cfc_hasani2022          "https://arxiv.org/pdf/2106.13898"
fetch liquid_s4_hasani2023    "https://arxiv.org/pdf/2209.12951"
fetch lrcssm_farsang2025      "https://arxiv.org/pdf/2505.21717"

# NCP (Lechner et al. 2020) has no arXiv preprint and is paywalled at Nature MI.
# TU Wien's institutional record (metadata only, no direct PDF link found) is
# at https://repositum.tuwien.at/handle/20.500.12708/141225 -- check it or
# Nature MI directly with your institutional access; not auto-downloadable.

echo "done -> papers/pdfs/"
