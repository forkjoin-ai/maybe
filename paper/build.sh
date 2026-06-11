#!/usr/bin/env bash
# Build the paper from markdown to PDF via pandoc
# Requires: pandoc, pdflatex (or xelatex)
set -euo pipefail

cd "$(dirname "$0")"

echo "Building Buleyean Manifold paper..."

pandoc buleyean-manifold.md \
  -o buleyean-manifold.pdf \
  --pdf-engine=tectonic \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V documentclass=article \
  --number-sections \
  --toc \
  2>&1

echo "Done: paper/buleyean-manifold.pdf"
