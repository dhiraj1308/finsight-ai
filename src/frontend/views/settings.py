import streamlit as st


def render(client):
    """Render the application settings page."""

    st.title("Settings")
    st.caption("Manage your FinSight AI application settings.")

    st.subheader("Application")

    st.info(
        "FinSight AI is currently running in local development mode. "
        "Application data is stored locally on this machine."
    )

    st.subheader("Backend")

    if client is not None:
        st.success("Backend client configured.")
    else:
        st.warning("Backend client is not available.")

    st.subheader("Data")

    st.write(
        "Uploaded financial statements are processed by the local FinSight AI backend."
    )

    st.subheader("About")

    st.write("FinSight AI — Personal Finance Intelligence")