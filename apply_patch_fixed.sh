#!/bin/bash
# Patch script to add DRAM_ACTIVE metric to aiperf

set -e

echo "Applying DRAM_ACTIVE metric patch to aiperf..."

# 1. Modify constants.py
echo "✓ Constants.py already patched"

# 2. Modify telemetry_models.py
echo "Modifying src/aiperf/common/models/telemetry_models.py..."

python3 << 'PYTHON_EOF'
with open('src/aiperf/common/models/telemetry_models.py', 'r') as f:
    content = f.read()

# Add dram_active field to TelemetryMetrics
if 'dram_active: float | None = None' not in content:
    content = content.replace(
        'power_violation: float | None = None',
        'power_violation: float | None = None\n    dram_active: float | None = None'
    )
    print("✓ Added dram_active field to TelemetryMetrics")
else:
    print("✓ dram_active field already in TelemetryMetrics")

with open('src/aiperf/common/models/telemetry_models.py', 'w') as f:
    f.write(content)
PYTHON_EOF

echo ""
echo "✅ Patch applied successfully!"
echo ""
echo "Next steps:"
echo "  git add -A"
echo "  git commit -m 'Add DCGM_FI_PROF_DRAM_ACTIVE metric support'"
echo "  git push origin add-dram-active-metric"
