# Bugfix Requirements Document

## Introduction

The FinSight AI Streamlit frontend has a sidebar navigation bug where users must click a
sidebar section twice before the page actually changes. The sidebar contains 8 sections:
Dashboard, Upload, Transactions, Analytics, Forecast, Anomalies, Chat, and Settings.
The fix must make every section respond to exactly one click, without altering any other
behavior, session state keys, or visual design.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN a user clicks a sidebar section that is different from the currently active page
    THEN the system does NOT navigate to the new page on that click — the page remains
    unchanged after the first click.

1.2 WHEN a user clicks a sidebar section a second time (i.e., clicks the same target section
    that was ignored on the first click) THEN the system navigates to the requested page.

1.3 WHEN `st.session_state["page"]` contains the name of the previously active page and
    `st.radio` is called with `index=default_index` derived from that stale value THEN
    the radio widget's selection is overridden back to the old page during the first
    rerun, discarding the user's click.

### Expected Behavior (Correct)

2.1 WHEN a user clicks any of the 8 sidebar sections (Dashboard, Upload, Transactions,
    Analytics, Forecast, Anomalies, Chat, Settings) THEN the system SHALL navigate to
    that section on that single click without requiring a second click.

2.2 WHEN the `st.radio` widget returns a new selection THEN the system SHALL use that
    return value as the active page immediately, without re-imposing a stale index from
    session state.

2.3 WHEN the application first loads (no page in session state) THEN the system SHALL
    default to the Dashboard page (index 0) as before.

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the application rerenders for any reason (e.g., widget interaction on the current
    page) THEN the system SHALL CONTINUE TO display the currently active page rather than
    resetting to Dashboard.

3.2 WHEN `st.session_state["page"]` is written after navigation THEN the system SHALL
    CONTINUE TO store the name of the currently displayed page so that other components
    can read it.

3.3 WHEN any of the 8 page views are rendered THEN the system SHALL CONTINUE TO receive
    the same `APIClient` instance and behave identically to the pre-fix implementation.

3.4 WHEN the sidebar is displayed THEN the system SHALL CONTINUE TO show the same title,
    caption, divider, and radio navigation list with all 8 section names unchanged.

3.5 WHEN the backend API, database, transaction data, or any non-frontend module is
    involved THEN the system SHALL CONTINUE TO operate without any modification.
