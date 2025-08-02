# app/components/charts.py

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


# ----------------------------
# 1. Pie Chart: Confirmed vs Rejected
# ----------------------------
def decision_pie_chart(df: pd.DataFrame):
    if df.empty:
        st.warning("No data to plot.")
        return

    decision_counts = df["decision"].value_counts().reset_index()
    decision_counts.columns = ["Decision", "Count"]

    fig = px.pie(
        decision_counts,
        names="Decision",
        values="Count",
        title="🔍 Operator Decisions Breakdown",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# 2. Bar Chart: False Positives/Negatives by Label
# ----------------------------
def error_bar_chart(df: pd.DataFrame):
    if df.empty:
        st.warning("No data to plot.")
        return

    grouped = df.groupby("label").agg({
        "false_positive": "sum",
        "false_negative": "sum"
    }).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped["label"],
        y=grouped["false_positive"],
        name="False Positives",
        marker_color="orange"
    ))
    fig.add_trace(go.Bar(
        x=grouped["label"],
        y=grouped["false_negative"],
        name="False Negatives",
        marker_color="crimson"
    ))

    fig.update_layout(
        barmode="group",
        title="📊 Errors by Target Type",
        xaxis_title="Target Label",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# 3. Timeline Chart: Decisions Over Time
# ----------------------------
def decisions_over_time_chart(df: pd.DataFrame):
    if df.empty:
        st.warning("No data to plot.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    daily_decisions = df.groupby(["date", "decision"]).size().reset_index(name="count")

    fig = px.line(
        daily_decisions,
        x="date",
        y="count",
        color="decision",
        markers=True,
        title="📅 Decisions Over Time",
        color_discrete_map={"Confirm": "green", "Reject": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)


# ----------------------------
# 4. Confidence Distribution
# ----------------------------
def confidence_distribution_chart(df: pd.DataFrame):
    if df.empty:
        st.warning("No data to plot.")
        return

    fig = px.histogram(
        df,
        x="confidence",
        nbins=20,
        title="📈 Target Confidence Distribution",
        color="decision",
        marginal="box",
        opacity=0.7,
        color_discrete_map={"Confirm": "green", "Reject": "red"}
    )
    fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Frequency")
    st.plotly_chart(fig, use_container_width=True)
