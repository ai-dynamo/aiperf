#!/bin/bash
# Patch script to add DRAM_ACTIVE metric to aiperf

set -e

echo "Applying DRAM_ACTIVE metric patch to aiperf..."

# 1. Modify constants.py
echo "Modifying src/aiperf/gpu_telemetry/constants.py..."

python3 << 'PYTHON_EOF'
with open('src/aiperf/gpu_telemetry/constants.py', 'r') as f:
    content = f.read()

# Add to DCGM_TO_FIELD_MAPPING
if '"DCGM_FI_PROF_DRAM_ACTIVE"' not in content:
    content = content.replace(
        '"DCGM_FI_DEV_POWER_VIOLATION": "power_violation",',
        '"DCGM_FI_DEV_POWER_VIOLATION": "power_violation",\n    "DCGM_FI_PROF_DRAM_ACTIVE": "dram_active",'
    )
    print("✓ Added DRAM_ACTIVE to DCGM_TO_FIELD_MAPPING")
else:
    print("✓ DRAM_ACTIVE already in DCGM_TO_FIELD_MAPPING")

# Add to GPU_TELEMETRY_METRICS_CONFIG
if '"DRAM Active"' not in content:
    content = content.replace(
        '("Power Violation", "power_violation", MetricTimeUnit.MICROSECONDS),',
        '("Power Violation", "power_violation", MetricTimeUnit.MICROSECONDS),\n    ("DRAM Active", "dram_active", GenericMetricUnit.PERCENT),'
    )
    print("✓ Added DRAM Active to GPU_TELEMETRY_METRICS_CONFIG")
else:
    print("✓ DRAM Active already in GPU_TELEMETRY_METRICS_CONFIG")

with open('src/aiperf/gpu_telemetry/constants.py', 'w') as f:
    f.write(content)
PYTHON_EOF

# 2. Modify models.py
echo "Modifying src/aiperf/common/models.py..."

python3 << 'PYTHON_EOF'
with open('src/aiperf/common/models.py', 'r') as f:
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

with open('src/aiperf/common/models.py', 'w') as f:
    f.write(content)
PYTHON_EOF

echo ""
echo "✅ Patch applied successfully!"
echo ""
echo "Next steps:"
echo "  git add -A"
echo "  git commit -m 'Add DCGM_FI_PROF_DRAM_ACTIVE metric support'"
echo "  git push origin add-dram-active-metric"
