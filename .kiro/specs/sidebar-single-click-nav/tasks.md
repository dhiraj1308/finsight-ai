# Implementation Plan

## Overview

Fix the sidebar single-click navigation bug in `src/frontend/app.py` by switching `st.radio` to use a `key=` parameter so Streamlit manages widget state internally, eliminating the stale `default_index` override that discards the user's first click.

## Task Dependency Graph

```json
{
  "waves": [
    { "wave": 1, "tasks": ["1"] },
    { "wave": 2, "tasks": ["2"] },
    { "wave": 3, "tasks": ["3"] },
    { "wave": 4, "tasks": ["4"] }
  ]
}
```

## Tasks

- [x] 1. Write bug condition exploration test
  - **Property 1: Bug Condition** - Stale Index Overrides Radio Selection
  - **CRITICAL**: This test MUST FAIL on unfixed code — failure confirms the bug exists
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: This test encodes the expected behavior — it will validate the fix when it passes after implementation
  - **GOAL**: Surface counterexamples that demonstrate `default_index` derived from stale `session_state["page"]` discards the user's first click
  - **Scoped PBT Approach**: For each page P_new ≠ P_current, simulate: session_state["page"] = P_current → call `main()` → assert rendered page == P_new
  - Bug Condition from design: `st.session_state["page"]` holds an old page name; `default_index` is computed from that stale value before `st.radio` is called; the radio widget's selection is overridden back to the old page during the first rerun
  - Test all 8 pages as P_current, and for each, assert that selecting any different P_new actually navigates there in one rerun
  - The test assertion should match Expected Behavior 2.1 and 2.2 from bugfix.md: one click navigates immediately; the radio return value is used directly without re-imposing stale index
  - Run test on UNFIXED code
  - **EXPECTED OUTCOME**: Test FAILS (this is correct — it proves the bug exists)
  - Document counterexamples found, e.g. "session_state['page']='Dashboard', clicking 'Upload' still renders Dashboard after one rerun"
  - Mark task complete when test is written, run, and failure is documented
  - _Requirements: 1.1, 1.2, 1.3_

- [-] 2. Write preservation property tests (BEFORE implementing fix)
  - **Property 2: Preservation** - Sidebar, Session State, and Page View Behavior Unchanged
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs (cases where no navigation click occurs):
    - Observe: on rerender without a navigation change, the currently active page continues to display (requirement 3.1)
    - Observe: after any navigation, `session_state["page"]` holds the name of the displayed page (requirement 3.2)
    - Observe: each view's `render(client)` receives the same `APIClient` instance (requirement 3.3)
    - Observe: sidebar always shows title "💰 FinSight AI", caption, divider, and radio with all 8 page names (requirement 3.4)
    - Observe: on first load with no `page` key in session_state, Dashboard is shown (requirement 2.3)
  - Write property-based tests capturing these observed behavior patterns:
    - For all valid pages P, re-rendering without a click keeps the active page as P
    - For all valid pages P, after navigation session_state["page"] == P
    - For all 8 page names, sidebar radio options contain exactly that name
    - On first load (no `page` key), default_index == 0 (Dashboard)
  - Verify all tests PASS on UNFIXED code
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 2.3, 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 3. Fix sidebar single-click navigation in src/frontend/app.py

  - [~] 3.1 Implement the fix
    - Add a `key="nav_radio"` parameter to the `st.radio` call so Streamlit manages widget state internally
    - Remove the `default_index` logic that reads `st.session_state["page"]` before calling `st.radio`
    - Only use `index=0` on first load when `"nav_radio"` key does not yet exist in session state (i.e., `st.session_state.get("nav_radio") is None`)
    - Derive the active page from the return value of `st.radio` directly — `selection = st.radio(...)` — rather than imposing any stale index
    - Keep `st.session_state["page"] = selection` write after the radio call so other components can still read the active page name (preserves 3.2)
    - Leave all view rendering logic, APIClient usage, sidebar title/caption/divider, and all non-frontend modules completely unchanged
    - _Bug_Condition: `st.session_state["page"]` holds stale page name → `default_index` overrides radio selection on first rerun (isBugCondition: `"page" in session_state AND default_index != radio's internal selection`)_
    - _Expected_Behavior: `selection = st.radio(..., key="nav_radio")` returns the user's chosen page; `session_state["page"]` is set to that value; the correct page view renders on the first click_
    - _Preservation: sidebar UI unchanged; session_state["page"] still updated; APIClient passed unchanged; Dashboard default on first load preserved_
    - _Requirements: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4, 3.5_

  - [~] 3.2 Verify bug condition exploration test now passes
    - **Property 1: Expected Behavior** - Stale Index Overrides Radio Selection
    - **IMPORTANT**: Re-run the SAME test from task 1 — do NOT write a new test
    - The test from task 1 encodes the expected behavior (one click → correct page rendered)
    - When this test passes, it confirms single-click navigation is working correctly
    - Run bug condition exploration test from step 1
    - **EXPECTED OUTCOME**: Test PASSES (confirms bug is fixed)
    - _Requirements: 2.1, 2.2_

  - [~] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Sidebar, Session State, and Page View Behavior Unchanged
    - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
    - Run preservation property tests from step 2
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)
    - Confirm rerender stability, session_state["page"] accuracy, APIClient integrity, sidebar structure, and Dashboard-on-first-load all still hold after fix

- [~] 4. Checkpoint — Ensure all tests pass
  - Run the full test suite and confirm all tests pass
  - Verify the fix works end-to-end by manually running the Streamlit app (`streamlit run src/frontend/app.py`) and clicking each of the 8 sidebar sections once to confirm immediate navigation
  - Ensure all tests pass; ask the user if any questions arise

## Notes

- Exploration test (task 1) is expected to **fail** on unfixed code — that is the correct outcome and confirms the bug exists. Do not fix the code or the test when it fails.
- Preservation tests (task 2) must **pass** on unfixed code — they capture the baseline behavior that must not regress.
- The fix is confined entirely to `src/frontend/app.py`; no other files should be modified.
- Manual verification: run `streamlit run src/frontend/app.py` and click each of the 8 sidebar sections once to confirm immediate single-click navigation.
