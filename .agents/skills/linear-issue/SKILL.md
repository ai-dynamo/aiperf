---
name: linear-issue
description: This skill should be used when the user asks to "create a linear issue", "add a linear issue", "track this in linear", "open a linear ticket", "log this in linear", "create a ticket", or mentions creating an issue for tracking work in Linear.
---

# Linear Issue Creator

Create Linear issues pre-configured with Todo status, assigned to the current user.

## Title Constraints

Linear generates git branch names in the format `username/TEAM-123-title-in-kebab-case`.
With ~18 characters consumed by the username prefix and issue identifier, keep the title
**50 characters or fewer** and avoid characters that are illegal in git branch names
(`~`, `^`, `:`, `?`, `*`, `[`, `\`, spaces, `..`).

Good titles (descriptive, short, branch-safe):
- `fix worker crash on empty response`
- `add latency percentile to report`
- `update auth middleware for compliance`

Bad titles:
- `Fix the worker service crash that happens when it receives an empty response` — too long
- `Fix: worker crash!` — contains special characters

## Workflow

### Step 1: Determine the team

Call `mcp__linear__list_teams` to retrieve available teams.

- One team: use it silently.
- Multiple teams: ask the user which team the issue belongs to.

### Step 2: Confirm the title

- If the context (current task, error, file, conversation) makes a suitable title obvious,
  propose one that satisfies the constraints above.
- Otherwise, ask the user for a short title.
- Always show the proposed title and wait for the user to confirm or revise before proceeding.

### Step 3: Create the issue

Call `mcp__linear__save_issue` with:

- `title` — confirmed title from Step 2
- `team` — team from Step 1
- `assignee` — `"me"`
- `state` — `"Todo"`

### Step 4: Offer to add to a project

Ask the user: "Do you want to add this to a Linear project?"

- If yes: call `mcp__linear__list_projects` with the team from Step 1, present the list,
  then call `mcp__linear__save_issue` with `id` set to the created issue identifier and
  `project` set to the chosen project name or ID.
- If no or no response: skip.

### Step 5: Offer to check out the branch

The `save_issue` response includes a `gitBranchName` field (e.g. `dbermudez/aip-123-title`).

Ask the user: "Do you want to check out the branch `<gitBranchName>`?"

- If yes: run `git checkout -b <gitBranchName>`.
- If no or no response: skip.

### Step 6: Confirm

Report the created issue identifier (e.g. `AIP-42`) and title to the user.
