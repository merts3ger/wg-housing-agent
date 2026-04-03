from langgraph.graph import StateGraph, START, END

from app.graph.state import HousingGraphState
from app.graph.nodes import (
    parse_listing_node,
    enrich_listing_node,
    budget_assessment_node,
    lifestyle_assessment_node,
    moderator_node,
)


def build_housing_graph():
    builder = StateGraph(HousingGraphState)

    builder.add_node("parse", parse_listing_node)
    builder.add_node("enrich", enrich_listing_node)
    builder.add_node("budget", budget_assessment_node)
    builder.add_node("lifestyle", lifestyle_assessment_node)
    builder.add_node("moderator", moderator_node)

    builder.add_edge(START, "parse")
    builder.add_edge("parse", "enrich")
    builder.add_edge("enrich", "budget")
    builder.add_edge("budget", "lifestyle")
    builder.add_edge("lifestyle", "moderator")
    builder.add_edge("moderator", END)

    return builder.compile()
