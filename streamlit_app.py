import os

import httpx
import streamlit as st

from app.schemas.profile import UserProfile

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Flatshare Fit Agent", page_icon="🏠", layout="wide")
st.title("🏠 Flatshare Fit Agent")

# ---------------------------------------------------------------------------
# STEP 1 — Profile setup
# ---------------------------------------------------------------------------
if "user_profile" not in st.session_state:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        with st.container(border=True):
            st.markdown("#### How it works")
            st.markdown(
                "This app evaluates flatshare listings against your personal profile. "
                "Paste a listing URL and it will extract the key details, enrich them "
                "with live location and commute data, and run two specialist agents — "
                "one on budget and value, one on lifestyle and daily fit — before a "
                "moderator synthesises a final verdict."
            )

    with col_right:
        with st.form("profile_form"):
            st.subheader("Your profile")

            budget_eur = st.number_input(
                "Max monthly budget (€, warm rent)", min_value=1, value=700, step=10
            )
            max_commute_minutes = st.number_input(
                "Max commute time (minutes)", min_value=1, max_value=180, value=30
            )
            commute_destination = st.text_input(
                "Commute destination (address or area)",
                placeholder="e.g. Christian-Albrechts-Platz 4, 24118 Kiel",
            )
            prefers_furnished = st.checkbox("Prefer furnished apartment", value=True)
            lifestyle_preference = st.selectbox(
                "Lifestyle preference", options=["quiet", "social", "urban"]
            )
            wants_park_nearby = st.checkbox("Park nearby is important", value=True)
            wants_swimming_pool_nearby = st.checkbox(
                "Swimming pool nearby is important", value=False
            )

            st.markdown("**Flatmate age preference** (optional — leave at 0 to skip)")
            col_min, col_max = st.columns(2)
            with col_min:
                age_min_raw = st.number_input(
                    "Minimum age", min_value=0, max_value=99, value=0, step=1
                )
            with col_max:
                age_max_raw = st.number_input(
                    "Maximum age", min_value=0, max_value=99, value=0, step=1
                )

            submitted = st.form_submit_button("Save profile & continue", type="primary")

        if submitted:
            if not commute_destination.strip():
                st.warning("Please enter a commute destination.")
                st.stop()
            st.session_state["user_profile"] = UserProfile(
                budget_eur=budget_eur,
                max_commute_minutes=max_commute_minutes,
                prefers_furnished=prefers_furnished,
                lifestyle_preference=lifestyle_preference,
                wants_park_nearby=wants_park_nearby,
                wants_swimming_pool_nearby=wants_swimming_pool_nearby,
                commute_destination=commute_destination,
                preferred_flatmate_age_min=age_min_raw if age_min_raw > 0 else None,
                preferred_flatmate_age_max=age_max_raw if age_max_raw > 0 else None,
            )
            st.rerun()

# ---------------------------------------------------------------------------
# STEP 2 — Evaluation
# ---------------------------------------------------------------------------
else:
    with st.sidebar:
        with st.container(border=True):
            st.markdown("**Budget & Value agent**")
            st.markdown(
                "Considers whether the warm rent fits your budget and how much headroom "
                "there is; value for space (price per m² and room size); furnishing status "
                "against your preference; and deposit size relative to monthly rent. "
                "Ignores amenities and location — those aren't financial signals."
            )

        with st.container(border=True):
            st.markdown("**Lifestyle & Daily Fit agent**")
            st.markdown(
                "Considers commute time as a daily quality-of-life factor, weighted by how "
                "precise the listing's location is; nearby parks and pools if you marked them "
                "as wanted; public transport mentioned in the ad; convenience amenities "
                "(dishwasher, washing machine, wifi); and the household's social character "
                "versus your lifestyle preference — a social WG counts against a 'quiet' "
                "preference even if it claims to respect privacy."
            )

    if st.button("Edit profile"):
        del st.session_state["user_profile"]
        st.rerun()

    st.divider()

    url = st.text_input("Listing URL", placeholder="https://www.wg-gesucht.de/...")

    if st.button("Evaluate", type="primary", disabled=not url):
        with st.spinner("Fetching and evaluating listing…"):
            try:
                response = httpx.post(
                    f"{BACKEND_URL}/evaluate-url-for-profile",
                    json={"url": url, "user_profile": st.session_state["user_profile"].model_dump()},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
            except httpx.ConnectError:
                st.error(f"Could not connect to backend at {BACKEND_URL}. Is the API server running?")
                st.stop()
            except httpx.HTTPStatusError as e:
                st.error(f"Evaluation failed ({e.response.status_code}): {e.response.text}")
                st.stop()
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

        # --- Verdict ---
        st.subheader("Verdict")
        if data.get("warnings"):
            st.warning(data["summary"])
        else:
            st.success(data["summary"])

        st.divider()

        # --- Agent scores ---
        col_b, col_l = st.columns(2)
        with col_b:
            st.metric(
                "Budget score",
                f"{budget['score']:.1f} / 10",
                delta="fits budget" if budget.get("fits_budget") else ("over budget" if budget.get("fits_budget") is False else None),
                delta_color="normal" if budget.get("fits_budget") else "inverse",
            )
        with col_l:
            st.metric(
                "Lifestyle score",
                f"{lifestyle['score']:.1f} / 10",
                delta="commute ok" if lifestyle.get("fits_commute") else ("commute long" if lifestyle.get("fits_commute") is False else None),
                delta_color="normal" if lifestyle.get("fits_commute") else "inverse",
            )

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

        if data.get("warnings"):
            st.divider()
            st.warning("\n".join(f"• {w}" for w in data["warnings"]))

        if data.get("questions_to_ask_host"):
            with st.expander("Questions to ask the host"):
                for q in data["questions_to_ask_host"]:
                    st.markdown(f"• {q}")
