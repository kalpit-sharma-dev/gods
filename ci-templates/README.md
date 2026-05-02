# Go CI Templates Starter Pack

This directory provides a portable **Option B** setup you can use to standardize
Go CI across repositories:

- reusable workflows repo layout (`workflow_call`)
- consumer repo wrapper examples
- governance template files (Dependabot, CODEOWNERS, PR template)

## Directory layout

- `reusable/`
  - Reusable workflow files intended to live in a dedicated templates
    repository, e.g. `your-org/ci-templates`.
- `wrappers/`
  - Example wrapper workflows to copy into any Go project that consumes the
    reusable templates.
- `templates/`
  - Governance template files (`CODEOWNERS`, PR template, Dependabot).

## Recommended rollout model

1. Create a dedicated repo, e.g. `your-org/ci-templates`.
2. Copy `reusable/*.yml` into that repo under `.github/workflows/`.
3. In each Go project, copy `wrappers/*.yml` into `.github/workflows/`.
4. Copy desired files from `templates/` into project `.github/`.
5. Update `uses:` references in wrapper workflows to point at your templates repo.

Example:

```yaml
jobs:
  pr-checks:
    uses: your-org/ci-templates/.github/workflows/go-ci-reusable.yml@main
```

## Per-project values to customize

You should tune these for each repository:

- coverage threshold (`min_coverage`)
- benchmark packages/pattern/allowed regression
- gosec exclusions (keep minimal)
- path filters and `paths-ignore`
- CODEOWNERS teams
- PR template checklist language

See:

- `.github/workflows/workflow-options.md` in the consumer project

## Branch protection recommendation

In each consumer repository, require:

- `pr-checks`

Optional but recommended:

- require up-to-date branch before merge
- require at least one approval
- restrict force-pushes on protected branches
