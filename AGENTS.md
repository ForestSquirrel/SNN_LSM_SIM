# AGENTS

## 0. Scope

These guidelines are **only applicable** when:

1. The prompt explicitly asks for **Doxygen / documentation work** in this repository, **or**
2. The prompt requires **modifying the codebase**, in which case:
   - First perform the requested code changes.
   - Then **apply these Doxygen rules** to all newly added or modified declarations.

If a prompt is unrelated to documentation and does not modify code, this file can be ignored.

---

## 1. Allowed Files and Safety Rules

When applying these documentation rules:

- Work **only** with files:
  - Extensions: `.h`, `.hpp`, `.cuh`
  - Locations: repository root, `core/`, `help/`
- **Do not touch**:
  - `kernel.cu` in the repository root (no edits, no comments, nothing)
- Prefer **minimal diffs**:
  - Only add or adjust comments needed for Doxygen
  - Do **not** change code behavior, signatures, or formatting unnecessarily
- Changes must be delivered via **Pull Request**, never by direct push

---

## 2. Documentation Goals

When these rules are in effect, the objectives are:

1. **Check for missing Doxygen comments** for:
   - Functions (free / member)
   - Classes / structs / enums
   - Templates
2. If comments are missing or clearly incomplete:
   - Add a Doxygen comment block directly above the corresponding declaration
3. Ensure comments are compatible with **doxide**:
   - **Never use** `@brief` (not supported)
   - Use only supported commands listed in Section 4
4. Whenever possible, assign entities to an existing **Doxygen group** using `@ingroup`
   - Use group names defined in `doxide.yaml`, such as:
     - `solvers`
     - `utilities`
     - `network`
       - `propagators`
         - `sparse_propagators`
         - `deprecated`
     - `trackers`
     - `datasets`
     - `lsm_utils`

## 3. Comment Style and Grouping Rules

### 3.1 General Style

- Place the Doxygen block **immediately above** the entity declaration.
- Use standard C++-style block comments:

  ```cpp
  /**
   * One-sentence description of what this does.
   *
   * @param ...
   * @return ...
   * @ingroup ...
   */
Prefer a concise, technical description.

Do not restate obvious details from the name; focus on purpose and behavior.

### 3.2 @ingroup Rules
If the entity clearly belongs to one of the existing groups in `doxide.yaml`:
- Add an `@ingroup <group_name>` as the last line of the Doxygen block.

If not:
- Do not invent or add a new `@ingroup`.
- Do not modify or remove existing `@ingroup` tags anywhere.

Instead, record this entity as a candidate for a new or refined group in `group_suggestions.txt` (see Section 5).

Example:

```cpp
/**
 * Computes the next state of the neuron using an Euler step.
 *
 * @param[in] dt Time step size.
 * @return Updated membrane potential.
 * @ingroup solvers
 */
double step(double dt);
```

## 4. Allowed doxide Commands and Meanings

Only the following commands should be used when writing or updating Doxygen comments
for this repository. Descriptions are given for reference.

`@param name`, `@param[in] name`, `@param[out] name`, `@param[in,out] name`
Document a function parameter name. The text that follows (usually the next paragraph or line) describes the parameter.
The optional `[in]`, `[out]`, `[in,out]` qualifiers indicate whether the parameter is used as input (default), output, or both.

`@tparam`
Document a template parameter. The following text describes what the template parameter represents.

`@return`
Document the return value of a function. The following text explains what is returned.

`@pre`, `@post`
Document preconditions (`@pre`) or postconditions (`@post`). The following text describes the condition that must hold before or after the call.

`@throw name`
Document a possible exception named name. The following text describes when or why this exception is thrown.

`@see`
Add a "see also" reference section. The text can contain Markdown (including links) pointing to relevant functions, types, or documentation.

`@anchor name`
Define an anchor called name that can be linked using Markdown syntax: `[text](#name)`.

`@ingroup name`
Add the documented entity to a Doxygen group name. Groups must come from doxide.yaml. Do not invent new names here; instead use `group_suggestions.txt` as described below.

## 5. `group_suggestions.tx`t Rules
If any part of the codebase does not clearly fit into existing groups from doxide.yaml:

Do not add an @ingroup tag for that entity.

Prepare or update a file named:
`group_suggestions.txt`

In this file, list suggested groups to be added using the following pattern:

```txt
- name: <group name>
title: <group title>
description: <group description>
at: <file_name1> | member: <class/function/enum/etc name> | line: <line number in file>
at: <file_name2> | member: <class/function/enum/etc name> | line: <line number in file>
...
at: <file_nameN> | member: <class/function/enum/etc name> | line: <line number in file>

<empty line>

<next group if needed>
```

Include `group_suggestions.txt` in the PR along with the documentation changes.

## 6. Workflow Guidance
### 6.1 When the prompt is explicitly about Doxygen / docs

1. Limit changes to `.h`, `.hpp`, `.cuh` in `root`, `core/`, `help/`.
2. Ignore `kernel.cu` completely.
3. Scan the relevant files for:
   - Missing Doxygen comments
   - Incomplete comments missing important tags (`@param`, `@return`, etc.)
4. Add or fix comments following Sections 3 and 4.
5. For entities not fitting any existing group:
   -  Leave them without `@ingroup`.
   - Record them in `group_suggestions.txt`.
6. Prepare a PR containing:
   - Updated header files
   - Optional `group_suggestions.txt` if any suggestions exist.

### 6.2 When the prompt is about modifying the codebase
1. Perform the requested code changes first (respecting file and safety rules).
2. Identify all new or modified declarations in `.h`, `.hpp`, `.cuh` within `root`, `core/`, `help/`.
3. Apply the same documentation workflow from Section 6.1 to those changed areas:
4. Add or update Doxygen comments:
   - Ensure proper `@param`, `@tparam`, `@return`, etc.
   - Add `@ingroup` only if it matches an existing group.
   - Add new group candidates to group_suggestions.txt if needed.
   - Include both the code changes and documentation updates (plus `group_suggestions.txt` if present) in a single PR.

## 7. Global Constraints
Never touch `kernel.cu`.

Never modify or remove existing `@ingroup` tags.

Never introduce `@brief`.

Never push directly; always assume changes go through a Pull Request.

Documentation changes must not alter semantics or behavior of the code.
