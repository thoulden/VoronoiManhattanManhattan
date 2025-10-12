
# Manhattan L1 Voronoi (Poster Generator)

Generate a poster-ready **Voronoi diagram under the Manhattan (L1) distance** for any city.
Default example: Manhattan using subway stations / cafés / bagel shops, etc. Data from OpenStreetMap.

<img alt="example" src="./.readme/example.png" width="600"/>

## Quick start

```bash
# Clone and set up
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Generate an SVG (default: Manhattan + subway stations)
python src/voronoi_l1.py \
  --city "Manhattan, New York, USA" \
  --query 'railway=station' \
  --grid-res 20 \
  --figsize "24x36" \
  --out "out/manhattan_voronoi_L1.svg"
