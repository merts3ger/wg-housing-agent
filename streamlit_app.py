import streamlit as st
from app.services.evaluator import evaluate_url_for_default_profile

st.set_page_config(page_title="Housing Agent", page_icon="🏠", layout="centered")
st.title("🏠 Housing Agent")
st.caption("Paste a listing URL to get a full evaluation.")

url = st.text_input("WG-Gesucht URL", placeholder="https://www.wg-gesucht.de/...")

if st.button("Evaluate", type="primary", disabled=not url):
    with st.spinner("Fetching and evaluating listing…"):
        try:
            result = evaluate_url_for_default_profile(url)
            data = result.model_dump()
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()

    snap = data["listing_snapshot"]
    budget = data["budget_assessment"]
    lifestyle = data["lifestyle_assessment"]

    # --- Listing snapshot ---
    st.subheader(snap["title"])
    col1, col2, col3 = st.columns(3)
    col1.metric("Warm rent", f"{snap['warm_rent']} €" if snap["warm_rent"] else "—")
    col2.metric("Room size", f"{snap['room_size_sqm']} m²" if snap["room_size_sqm"] else "—")
    col3.metric("Commute", f"{snap['commute_minutes']} min" if snap["commute_minutes"] else "—")

    details = []
    if snap["neighborhood"]:
        details.append(f"📍 {snap['neighborhood']}")
    if snap["available_from"]:
        details.append(f"📅 Available from {snap['available_from']}")
    if snap["furnishing_status"]:
        details.append(f"🛋 {snap['furnishing_status'].replace('_', ' ').capitalize()}")
    if details:
        st.caption("  ·  ".join(details))

    st.divider()

    # --- Agent scores ---
    col_b, col_l = st.columns(2)
    with col_b:
        st.metric("Budget score", f"{budget['score']:.1f} / 10",
                  delta="fits budget" if budget.get("fits_budget") else ("over budget" if budget.get("fits_budget") is False else None),
                  delta_color="normal" if budget.get("fits_budget") else "inverse")
    with col_l:
        st.metric("Lifestyle score", f"{lifestyle['score']:.1f} / 10",
                  delta="commute ok" if lifestyle.get("fits_commute") else ("commute long" if lifestyle.get("fits_commute") is False else None),
                  delta_color="normal" if lifestyle.get("fits_commute") else "inverse")

    # --- Pros / cons ---
    col_b2, col_l2 = st.columns(2)
    with col_b2:
        st.markdown("**Budget**")
        for p in budget.get("pros", []):
            st.markdown(f"✅ {p}")
        for c in budget.get("cons", []):
            st.markdown(f"❌ {c}")
        if budget.get("notes"):
            st.caption(budget["notes"])
    with col_l2:
        st.markdown("**Lifestyle**")
        for p in lifestyle.get("pros", []):
            st.markdown(f"✅ {p}")
        for c in lifestyle.get("cons", []):
            st.markdown(f"❌ {c}")
        if lifestyle.get("notes"):
            st.caption(lifestyle["notes"])

    st.divider()

    # --- Verdict ---
    st.subheader("Verdict")
    st.info(data["summary"])

    if data.get("warnings"):
        st.warning("\n".join(f"• {w}" for w in data["warnings"]))

    if data.get("questions_to_ask_host"):
        with st.expander("Questions to ask the host"):
            for q in data["questions_to_ask_host"]:
                st.markdown(f"• {q}")
