# Reusable workflow customization guide

This repository uses reusable workflows (`workflow_call`) so other Go projects can adopt the same CI model with project-specific inputs.

## Reusable workflows

- `.github/workflows/go-ci-reusable.yml`
- `.github/workflows/go-security-reusable.yml`
- `.github/workflows/go-performance-reusable.yml`

## Required project customizations

### 1) Coverage threshold

Set per project based on actual baseline:

- input: `min_coverage`
- default currently wired in wrappers: `30`

### 2) Benchmark workflow

Customize all of:

- `benchmark_packages` (space-separated package list passed to `go test`)
- `benchmark_pattern` (bench names regex)
- `max_regression_pct`

Defaults in this repo are tuned for `gods` and should be changed in other projects.

### 3) Security rules (`gosec`)

Customize `gosec_excludes`:

- default: `G115,G304,G306,G404`
- keep this list as small as possible; reduce exclusions over time.

### 4) Paths filters

In `.github/workflows/go-pr-checks.yml`, update:

- `change_filters` (the paths considered CI-relevant)

In `.github/workflows/go-push.yml`, update:

- `paths_ignore`

### 5) CODEOWNERS

Replace owners and folder mappings in `.github/CODEOWNERS` for each repository.

### 6) PR template

Adapt `.github/pull_request_template.md` checklist and sections to your team process.

## Example usage in another repo

```yaml
jobs:
  ci:
    uses: your-org/your-repo/.github/workflows/go-ci-reusable.yml@main
    with:
      min_coverage: 45
      run_gosec: true
      gosec_excludes: G304
      pr_mode: true
      run_go_ci: true
```

