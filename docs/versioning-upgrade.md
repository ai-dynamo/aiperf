<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Versioning & Upgrade Guide

This guide explains AIPerf's versioning scheme, compatibility guarantees, and upgrade procedures.

## Table of Contents

- [Versioning Scheme](#versioning-scheme)
- [Compatibility Policy](#compatibility-policy)
- [Upgrade Procedures](#upgrade-procedures)
- [Breaking Changes](#breaking-changes)
- [Deprecation Policy](#deprecation-policy)
- [Version History](#version-history)

## Versioning Scheme

AIPerf follows [Semantic Versioning 2.0.0](https://semver.org/):

```
MAJOR.MINOR.PATCH
```

### Version Components

**MAJOR** (e.g., 1.x.x → 2.x.x)
- Breaking changes to CLI interface
- Removal of deprecated features
- Incompatible changes to output formats
- Major architectural changes

**MINOR** (e.g., 1.0.x → 1.1.x)
- New features (backward compatible)
- New CLI options
- New metrics
- New plugins
- Performance improvements

**PATCH** (e.g., 1.0.0 → 1.0.1)
- Bug fixes
- Documentation updates
- Minor performance improvements
- Security patches

### Pre-Release Versions

Development versions use suffixes:
- **Alpha**: `1.0.0-alpha.1` - Early development, unstable
- **Beta**: `1.0.0-beta.1` - Feature complete, testing phase
- **RC**: `1.0.0-rc.1` - Release candidate, final testing

## Compatibility Policy

### CLI Compatibility

**Backward Compatibility (Within MAJOR version):**
- Existing CLI commands continue to work
- New options are additive
- Defaults may change (with deprecation notice)
- Output format changes are considered breaking

**Example:**
```bash
# Version 1.0.0
aiperf profile --model M --url U --concurrency 10

# Version 1.5.0 (backward compatible)
aiperf profile --model M --url U --concurrency 10  # Still works
aiperf profile --model M --url U --concurrency 10 --new-option value  # New feature
```

### Output Format Compatibility

**Stable Formats:**
- CSV structure (column names, types)
- JSON schema (field names, types)
- JSONL record structure

**Changes Requiring MAJOR Version:**
- Removing fields from output
- Changing field types
- Renaming fields
- Changing metric calculation formulas

**Backward Compatible Changes (MINOR):**
- Adding new fields
- Adding new metrics
- New output formats

### Python API Compatibility

If using AIPerf as a library:
- Public APIs maintain compatibility within MAJOR version
- Internal APIs (prefixed with `_`) may change anytime
- Plugin interfaces are considered public APIs

### Plugin Compatibility

**Plugin Registry (`plugins.yaml`):**
- Schema version indicates compatibility
- Schema changes require MAJOR version bump
- Existing plugins continue to work within MAJOR version

**Custom Plugins:**
- Protocol interfaces are stable within MAJOR version
- Metadata schemas are stable within MAJOR version

## Upgrade Procedures

### Checking Your Version

```bash
# Current version
aiperf --version

# Available versions
pip index versions aiperf
```

### Upgrading via pip

**Patch Upgrade (Recommended):**
```bash
# Upgrade to latest patch within current minor version
pip install --upgrade 'aiperf~=1.0.0'

# Example: If you have 1.0.0, this upgrades to 1.0.x (latest patch)
```

**Minor Upgrade:**
```bash
# Upgrade to latest minor version within current major version
pip install --upgrade 'aiperf>=1.0.0,<2.0.0'

# Or simply
pip install --upgrade aiperf
```

**Major Upgrade (Read changelog first):**
```bash
# Specify exact major version
pip install --upgrade 'aiperf>=2.0.0,<3.0.0'
```

### Upgrading from Source

```bash
cd aiperf
git fetch --tags
git checkout v1.5.0  # Replace with desired version
make install
```

### Docker Upgrades

```bash
# Pull specific version
docker pull aidynamo/aiperf:1.5.0

# Or pull latest
docker pull aidynamo/aiperf:latest
```

### Kubernetes Upgrades

Update image version in your manifests:

```yaml
spec:
  containers:
  - name: aiperf
    image: aidynamo/aiperf:1.5.0  # Update version
```

Apply changes:
```bash
kubectl apply -f aiperf-job.yaml
```

## Breaking Changes

### How Breaking Changes Are Communicated

1. **Deprecation Notice** (at least one MINOR version before removal)
2. **Migration Guide** (in changelog and docs)
3. **Warning Messages** (when using deprecated features)
4. **CHANGELOG.md** (detailed list of breaking changes)

### Example Migration Path

**Deprecated in 1.5.0:**
```bash
# Old way (still works in 1.5.0)
aiperf profile --old-option value

# Warning: --old-option is deprecated and will be removed in 2.0.0.
# Use --new-option instead.
```

**Removed in 2.0.0:**
```bash
# Old way (ERROR in 2.0.0)
aiperf profile --old-option value
# Error: Unknown option '--old-option'. Use '--new-option' instead.

# New way
aiperf profile --new-option value
```

### Common Breaking Changes

**1. CLI Option Renames**
```bash
# Before (deprecated)
aiperf profile --max-threads 8

# After
aiperf profile --workers-max 8
```

**2. Output Format Changes**
```bash
# Before: Field name 'ttft'
{"ttft": 50.0}

# After: Field name 'time_to_first_token'
{"time_to_first_token": 50.0}
```

**3. Default Value Changes**
```bash
# Before: Default --workers-max=16
aiperf profile  # Used 16 workers

# After: Default --workers-max=32
aiperf profile  # Uses 32 workers
```

## Deprecation Policy

### Deprecation Timeline

1. **Announcement** (MINOR release)
   - Feature marked deprecated
   - Warning messages added
   - Migration guide published

2. **Deprecation Period** (minimum 1 MINOR release)
   - Feature still works
   - Warnings displayed
   - Users have time to migrate

3. **Removal** (MAJOR release)
   - Feature removed
   - Error message if used
   - Migration guide in docs

### Example Timeline

```
v1.3.0 - Feature working normally
v1.4.0 - Feature deprecated (warnings)
v1.5.0 - Feature still deprecated (warnings)
v2.0.0 - Feature removed (errors)
```

### Checking for Deprecated Features

Run with warnings enabled:
```bash
export PYTHONWARNINGS=default
aiperf profile ...
```

Look for deprecation warnings in output:
```
DeprecationWarning: --old-option is deprecated in 1.5.0
and will be removed in 2.0.0. Use --new-option instead.
```

## Version History

### Version 1.x

**1.0.0** (Initial Release)
- Core benchmarking features
- Concurrency and request-rate modes
- Basic metrics (TTFT, ITL, throughput)
- OpenAI-compatible endpoints

**1.1.0** (Minor Update)
- Added GPU telemetry support
- Added server metrics collection
- New UI types (dashboard, simple, none)
- Performance improvements

**1.2.0** (Feature Release)
- Multi-turn conversation support
- ShareGPT dataset integration
- Goodput metrics
- HTTP trace metrics

**1.3.0** (Enhancement Release)
- Plugin system
- Custom dataset loaders
- Trace replay mode
- Timeslice analysis

**1.4.0** (Optimization Release)
- Worker scaling improvements
- Memory usage optimizations
- Connection pooling enhancements
- New arrival patterns (gamma, concurrency burst)

**1.5.0** (Current)
- User-centric timing mode
- Prefill concurrency limits
- Enhanced plotting
- Performance tuning improvements

### Migration Guides

#### Migrating from GenAI-Perf

See **[Migrating from GenAI-Perf](migrating.md)** for complete guide.

**Key Changes:**
- `--max-threads` → `--workers-max`
- Remove `--` passthrough flag (no longer needed)
- `inputs.json` format changed (see migration guide)

#### Upgrading from 0.x to 1.x

**Breaking Changes:**
1. CLI option renames
2. Output format changes
3. Configuration file format

**Migration Steps:**
```bash
# 1. Update CLI options
# Before (0.x)
aiperf benchmark --threads 8

# After (1.x)
aiperf profile --workers-max 8

# 2. Update output parsing
# Field names changed in JSON exports

# 3. Review changelog
cat CHANGELOG.md
```

## Best Practices

### Before Upgrading

1. **Read the Changelog**
   ```bash
   # View changes for target version
   curl https://raw.githubusercontent.com/ai-dynamo/aiperf/v1.5.0/CHANGELOG.md
   ```

2. **Test in Non-Production**
   ```bash
   # Install new version in test environment
   pip install aiperf==1.5.0

   # Run test benchmarks
   aiperf profile --model test-model ...
   ```

3. **Backup Results**
   ```bash
   # Save current results for comparison
   cp -r artifacts/ artifacts_backup/
   ```

4. **Check Compatibility**
   ```bash
   # Test your scripts/automation
   ./run_benchmarks.sh
   ```

### After Upgrading

1. **Verify Installation**
   ```bash
   aiperf --version
   pip show aiperf
   ```

2. **Run Baseline Benchmark**
   ```bash
   # Same config as before upgrade
   aiperf profile --random-seed 42 ...
   ```

3. **Compare Results**
   ```bash
   # Ensure metrics are consistent
   aiperf plot --paths artifacts_old/ artifacts_new/
   ```

4. **Update Documentation**
   - Update internal docs with new version
   - Update CI/CD configs
   - Update Docker images

### Pinning Versions

For reproducible environments:

**pip:**
```bash
# requirements.txt
aiperf==1.5.0
```

**Docker:**
```dockerfile
FROM aidynamo/aiperf:1.5.0
```

**Kubernetes:**
```yaml
image: aidynamo/aiperf:1.5.0
```

## Troubleshooting Upgrades

### Version Conflicts

```bash
# Error: Multiple versions installed
pip uninstall aiperf -y
pip install aiperf==1.5.0
```

### Broken Dependencies

```bash
# Update all dependencies
pip install --upgrade aiperf

# Or reinstall fresh
pip uninstall aiperf -y
pip install aiperf
```

### Incompatible Plugins

```bash
# Check plugin compatibility
aiperf plugins --validate

# Update custom plugins if needed
```

### Configuration Errors

```bash
# Validate config with new version
aiperf profile --help | grep your-option

# Check environment variables
env | grep AIPERF
```

## Getting Help

### Version-Specific Documentation

```bash
# View docs for specific version
# https://github.com/ai-dynamo/aiperf/tree/v1.5.0/docs
```

### Reporting Issues

When reporting upgrade issues, include:
- Previous version
- New version
- Upgrade method (pip, Docker, source)
- Error messages
- Full command used

### Support Channels

- **GitHub Issues**: [https://github.com/ai-dynamo/aiperf/issues](https://github.com/ai-dynamo/aiperf/issues)
- **Discord**: [https://discord.gg/D92uqZRjCZ](https://discord.gg/D92uqZRjCZ)
- **Discussions**: [https://github.com/ai-dynamo/aiperf/discussions](https://github.com/ai-dynamo/aiperf/discussions)

## See Also

- **[Getting Started](getting-started.md)** - Installation guide
- **[Troubleshooting](troubleshooting.md)** - Common issues
- **[Migrating from GenAI-Perf](migrating.md)** - Migration guide
- **[CHANGELOG](../CHANGELOG.md)** - Detailed version history
- **[Contributing](../CONTRIBUTING.md)** - Development guidelines
