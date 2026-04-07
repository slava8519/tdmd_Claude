# Prompts & Roles for Claude Code

> This folder contains the prompts the human (or Claude itself) loads to switch context
> for different kinds of work on TDMD.

## How prompts work in this project

`CLAUDE.md` at the root sets the **baseline** rules: how to commit, test, document, when to ask. It's loaded automatically by Claude Code at session start.

The files here are **role overlays** and **workflow recipes**. The human (or Claude) loads one when starting a specific task, by saying e.g.:

> "Acting as `physicist-validator`, design VerifyLab cases for the M1 Morse implementation."

> "Use the `add-kernel` workflow to implement the EAM force kernel."

## Folder layout

```
prompts/
├── README.md                    # this file
├── roles/                       # who you are right now
│   ├── architect.md
│   ├── implementer.md
│   ├── reviewer.md
│   ├── test-engineer.md
│   ├── physicist-validator.md
│   └── doc-writer.md
├── workflows/                   # what you do, step by step
│   ├── start-new-milestone.md
│   ├── add-feature.md
│   ├── fix-bug.md
│   ├── add-kernel.md
│   ├── validate-against-lammps.md
│   ├── update-docs.md
│   └── investigate-performance.md
└── task-templates/              # fill-in-the-blank task descriptions
    ├── milestone-task.md
    ├── bug-report.md
    └── performance-investigation.md
```

## When to load which

**Roles** modify *how* you behave: tone, priorities, what you check before responding. Load one role at a time.

**Workflows** are step-by-step recipes for *what* to do. They are role-agnostic — you can run a workflow as any role.

**Task templates** are forms the human fills in to give Claude a precise spec.

## Defaults

If nothing is loaded explicitly, Claude operates as the `implementer` role with general project rules from `CLAUDE.md`.

## Stacking

You can stack a role with a workflow. E.g.:

> "As `reviewer`, run the `validate-against-lammps` workflow on this PR."

This means: take the role-specific priorities of `reviewer` (catch issues, suggest improvements) and apply them while following the steps in `validate-against-lammps`.

## When in doubt

If a prompt and `CLAUDE.md` conflict, **`CLAUDE.md` always wins.** Roles and workflows refine, they do not override the project rules.
