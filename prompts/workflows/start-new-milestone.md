# Workflow: Start a new milestone

Use this when transitioning from one milestone to the next.

## Steps

1. **Confirm the previous milestone is closed.**
   - All exit criteria checked off in `docs/03-roadmap/milestones.md`.
   - All VerifyLab cases for that milestone are green.
   - `CHANGELOG.md` has an entry "M<N> complete" with date.
   - Human has signed off in chat.

2. **Read the new milestone spec.**
   - `docs/03-roadmap/milestones.md` (the relevant section).
   - Any additional milestone-specific notes if they exist.

3. **List the work.**
   - Break the milestone into 5–15 tasks.
   - Each task should be independently committable.
   - Order by dependency.
   - Post the list in chat for human review.

4. **Create a branch.**
   - `git checkout -b m<N>-<short-name>`
   - Example: `git checkout -b m1-reference-md`

5. **Pick the first task.** Run the `add-feature` workflow on it.

6. **At end of milestone:**
   - Run all unit tests, all VerifyLab cases relevant to this milestone.
   - Update `docs/03-roadmap/milestones.md` with status `✅`.
   - Update `CHANGELOG.md`.
   - Open a PR titled `M<N>: <name>`.
   - Wait for human review and merge.

## Output

After step 3 you should post a message like:

> "Milestone M<N> plan: <name>. Here are the <K> tasks I propose, in order: ... Shall I start with task 1?"
