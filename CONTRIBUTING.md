# Contributing to TDMD

> TDMD is currently developed by a single human + Claude Code (AI agent).
> External contributions will be welcomed once we reach M1+.

## For the human developer

1. Read `CLAUDE.md` to understand the rules of the project.
2. Read `docs/00-vision.md` and `docs/03-roadmap/milestones.md` to understand where we are.
3. Create branches per `CLAUDE.md` §8.
4. Use `prompts/` to switch Claude into the right role for each task.
5. Always run `./scripts/build.sh` and `./scripts/run-tests.sh` before committing.
6. Update docs and `CHANGELOG.md` for any user-visible change.

## For Claude Code (AI agent)

You are bound by the rules in `CLAUDE.md`. They take precedence over anything in this file or in role prompts. The short version:

- Never break the build.
- Never commit without tests + docs.
- Validate against LAMMPS as early as possible.
- Ask the human when in doubt about physics or architecture.
- Prefer boring readable code over clever fast code (until M7).

## Pull request checklist

Before opening a PR (or before merging a branch):

- [ ] Build passes (`./scripts/build.sh`).
- [ ] Tests pass (`./scripts/run-tests.sh`).
- [ ] VerifyLab green for any physics-touching change.
- [ ] Documentation updated.
- [ ] `CHANGELOG.md` updated under "Unreleased" if user-visible.
- [ ] Commit messages follow `<scope>: <imperative summary>`.
- [ ] No leftover TODO without an issue link.
- [ ] No new third-party dependency without an ADR.

## Code of conduct

Be honest, technical, and kind. That's it.
