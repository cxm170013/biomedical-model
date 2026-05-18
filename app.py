import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from synthetic_data import generate_subject_metadata, generate_model_outputs


st.set_page_config(
    page_title="Biomedical Model Explorer",
    layout="wide"
)

st.title("Biomedical Model Explorer")
st.caption("Prototype for exploring evolving multimodal biomedical ML models")


metadata = generate_subject_metadata()

st.sidebar.header("Controls")

model_version = st.sidebar.selectbox(
    "Model version",
    ["v1", "v2"]
)

view_mode = st.sidebar.radio(
    "User view",
    ["Researcher View", "Clinician View"]
)

st.sidebar.subheader("Modality Sensitivity")

use_speech = st.sidebar.checkbox("Use Speech", value=True)
use_wearable = st.sidebar.checkbox("Use Wearable", value=True)
use_mri = st.sidebar.checkbox("Use MRI", value=True)

enabled_modalities = []
if use_speech:
    enabled_modalities.append("speech")
if use_wearable:
    enabled_modalities.append("wearable")
if use_mri:
    enabled_modalities.append("mri")

if not enabled_modalities:
    st.sidebar.warning("At least one modality should be selected.")
    enabled_modalities = ["speech"]

review_threshold = st.sidebar.slider(
    "Confidence review threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.55,
    step=0.05
)

sensitivity_noise = st.sidebar.slider(
    "Sensitivity / noise level",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1
)

outputs, embedding_df, pca = generate_model_outputs(
    metadata,
    model_version=model_version,
    embedding_dim=128,
    enabled_modalities=enabled_modalities,
    sensitivity_noise=sensitivity_noise
)


st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Subjects", metadata["subject_id"].nunique())
col2.metric("Sessions", len(metadata))
col3.metric("Model Version", model_version)
col4.metric("Active Modalities", len(enabled_modalities))

st.subheader("Modality Availability")

availability = metadata[
    ["speech_available", "wearable_available", "mri_available"]
].mean().reset_index()

availability.columns = ["Modality", "Availability Rate"]
availability["Availability Rate"] *= 100

fig_availability = px.bar(
    availability,
    x="Modality",
    y="Availability Rate",
    title="Available Data by Modality"
)

st.plotly_chart(fig_availability, use_container_width=True)



st.header("2. Model Pipeline Overview")

model_description = {
    "v1": {
        "name": "CNN-based Multimodal Encoder",
        "summary": ""
    },
    "v2": {
        "name": "Transformer-based Multimodal Encoder",
        "summary": ""
    }
}

current_model = model_description[model_version]

st.subheader(f"Selected Model: {current_model['name']}")

st.markdown(current_model["summary"])

encoder_name = (
    "CNN-based Multimodal Encoder"
    if model_version == "v1"
    else "Transformer-based Multimodal Encoder"
)

col1, col2, col3 = st.columns(3)

col1.info("Input 1: Speech recordings or speech-derived features")
col2.info("Input 2: Wearable sensor signals")
col3.info("Input 3: MRI-derived imaging features")

st.code(
    f"""
Speech recordings / features      ┐
Wearable sensor data              ├── [ {encoder_name} ] ──> 128D Embedding
MRI images / derived features     ┘

    """,
    language="text"
)



# st.success(
#     ""
#     #"This allows researchers to compare evolving architectures while keeping the exploration workflow consistent."
# )



st.header("3. Model Output Explorer")

if view_mode == "Researcher View":
    st.subheader("From 128D Embeddings to 2D PCA Projection")

    st.markdown("""
    The model produces a **128-dimensional embedding** for each subject-session pair.
    PCA projects those embeddings into two dimensions for visualization.
    """)

    st.code(
        "Raw multimodal data → 128D model embedding → PCA → 2D visualization",
        language="text"
    )

    explained_variance = pca.explained_variance_ratio_

    col1, col2 = st.columns(2)
    col1.metric("PCA Component 1 Variance", f"{explained_variance[0] * 100:.1f}%")
    col2.metric("PCA Component 2 Variance", f"{explained_variance[1] * 100:.1f}%")

    selected_subject_for_embedding = st.selectbox(
        "Inspect raw 128D embedding for subject-session",
        outputs.apply(
            lambda row: f"{row['subject_id']} | Session {row['session']}",
            axis=1
        )
    )

    selected_index = outputs.apply(
        lambda row: f"{row['subject_id']} | Session {row['session']}",
        axis=1
    ).tolist().index(selected_subject_for_embedding)

    raw_embedding_preview = embedding_df.iloc[selected_index].head(20).reset_index()
    raw_embedding_preview.columns = ["Embedding Dimension", "Value"]

    st.write("First 20 dimensions of the 128D embedding:")
    st.dataframe(raw_embedding_preview, use_container_width=True)

else:
    st.subheader("Patient Similarity Map")

    st.markdown("""
    Each point represents one subject-session. Points closer together have more similar
    model representations. This view hides raw technical details and focuses on patterns,
    outliers, and cases that may need review.
    """)

fig_embedding = px.scatter(
    outputs,
    x="pca_1",
    y="pca_2",
    color="diagnosis",
    size="confidence",
    hover_data=[
        "subject_id",
        "session",
        "behavior_score",
        "missing_modalities",
        "confidence",
        "active_modalities"
    ],
    title=(
        "2D PCA Projection of 128D Model Embeddings"
        if view_mode == "Researcher View"
        else "Patient Similarity Map"
    ),
    labels={
        "pca_1": "PCA Component 1",
        "pca_2": "PCA Component 2",
        "diagnosis": "Diagnosis"
    }
)

st.plotly_chart(fig_embedding, use_container_width=True)


# st.header("4. Subject-Level Review and Session Comparison")

# selected_subject = st.selectbox(
#     "Select subject",
#     sorted(outputs["subject_id"].unique())
# )

# subject_data = outputs[outputs["subject_id"] == selected_subject]

# st.dataframe(subject_data, use_container_width=True)

# if len(subject_data) == 2:
#     s1 = subject_data[subject_data["session"] == 1].iloc[0]
#     s2 = subject_data[subject_data["session"] == 2].iloc[0]

#     pca_drift = np.sqrt(
#         (s2["pca_1"] - s1["pca_1"]) ** 2 +
#         (s2["pca_2"] - s1["pca_2"]) ** 2
#     )

#     score_change = abs(s2["behavior_score"] - s1["behavior_score"])
#     confidence_change = s2["confidence"] - s1["confidence"]

#     if pca_drift < 2:
#         stability_label = "Stable"
#     elif pca_drift < 5:
#         stability_label = "Moderate variability"
#     else:
#         stability_label = "High variability"

#     col1, col2 = st.columns(2)

#     col1.metric("Stability Status", stability_label)
#     col2.metric("Clinical Score Change", f"{score_change:.2f}")

# if view_mode == "Researcher View":
#     fig_subject = go.Figure()

#     fig_subject.add_trace(go.Scatter(
#         x=subject_data["pca_1"],
#         y=subject_data["pca_2"],
#         mode="markers+text",
#         text=["Session 1", "Session 2"],
#         textposition="top center",
#         marker=dict(size=14),
#         name=selected_subject
#     ))

#     fig_subject.add_annotation(
#         x=s2["pca_1"],
#         y=s2["pca_2"],
#         ax=s1["pca_1"],
#         ay=s1["pca_2"],
#         xref="x",
#         yref="y",
#         axref="x",
#         ayref="y",
#         showarrow=True,
#         arrowhead=3
#     )

#     fig_subject.update_layout(
#         title=f"Session Movement for {selected_subject}",
#         xaxis_title="PCA Component 1",
#         yaxis_title="PCA Component 2"
#     )

#     st.plotly_chart(fig_subject, use_container_width=True)

# else:
#     if pca_drift < 2:
#         stability_label = "Stable"
#         stability_message = (
#             "The subject appears relatively stable across sessions based on the model representation."
#         )
#     elif pca_drift < 5:
#         stability_label = "Moderate variability"
#         stability_message = (
#             "The subject shows some session-to-session change. This may be clinically meaningful "
#             "or may reflect variation in data quality."
#         )
#     else:
#         stability_label = "High variability"
#         stability_message = (
#             "The subject shows substantial session-to-session change. This case may benefit from closer review."
#         )

#     st.subheader("Clinical Stability Summary")

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Stability Status", stability_label)
#     col2.metric("Clinical Score Change", f"{score_change:.2f}")
#     col3.metric("Confidence Change", f"{confidence_change:.2f}")

#     st.info(stability_message)


st.header("4. Subject-Level Review and Session Comparison")

selected_subject = st.selectbox(
    "Select subject",
    sorted(outputs["subject_id"].unique())
)

subject_data = outputs[outputs["subject_id"] == selected_subject]

st.dataframe(subject_data, use_container_width=True)

if len(subject_data) == 2:
    s1 = subject_data[subject_data["session"] == 1].iloc[0]
    s2 = subject_data[subject_data["session"] == 2].iloc[0]

    pca_drift = np.sqrt(
        (s2["pca_1"] - s1["pca_1"]) ** 2 +
        (s2["pca_2"] - s1["pca_2"]) ** 2
    )

    score_change = abs(s2["behavior_score"] - s1["behavior_score"])
    confidence_change = s2["confidence"] - s1["confidence"]

    if view_mode == "Researcher View":
        col1, col2, col3 = st.columns(3)
        col1.metric("2D Session Drift", f"{pca_drift:.2f}")
        col2.metric("Clinical Score Change", f"{score_change:.2f}")
        col3.metric("Confidence Change", f"{confidence_change:.2f}")

        fig_subject = go.Figure()

        fig_subject.add_trace(go.Scatter(
            x=subject_data["pca_1"],
            y=subject_data["pca_2"],
            mode="markers+text",
            text=["Session 1", "Session 2"],
            textposition="top center",
            marker=dict(size=14),
            name=selected_subject
        ))

        fig_subject.add_annotation(
            x=s2["pca_1"],
            y=s2["pca_2"],
            ax=s1["pca_1"],
            ay=s1["pca_2"],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3
        )

        fig_subject.update_layout(
            title=f"Session Movement for {selected_subject}",
            xaxis_title="PCA Component 1",
            yaxis_title="PCA Component 2"
        )

        st.plotly_chart(fig_subject, use_container_width=True)

        st.info(
            "Technical summary: This panel shows how the subject's latent representation "
            "moves across sessions, along with clinical score and confidence changes."
        )

    else:
        #st.subheader("Clinical Stability Summary")

        if score_change < 5:
            stability_label = "Stable"
        elif score_change < 15:
            stability_label = "Moderate variability"
        else:
            stability_label = "High variability"

        s1_missing = s1["missing_modalities"]
        s2_missing = s2["missing_modalities"]

        if max(s1_missing, s2_missing) >= 2:
            reliability_label = "Low reliability"
            reliability_message = (
                "Two or more modalities are missing in at least one session. "
                "Interpret the session comparison with caution."
            )
        elif s1_missing != s2_missing:
            reliability_label = "Use caution"
            reliability_message = (
                "The two sessions have different modality availability. "
                "Some apparent change may reflect missing data rather than true clinical change."
            )
        else:
            reliability_label = "Comparable sessions"
            reliability_message = (
                "Both sessions have similar modality availability, so the clinical comparison is more reliable."
            )

        # col1, col2, col3 = st.columns(3)
        # col1.metric("Clinical Stability", stability_label)
        # col2.metric("Clinical Score Change", f"{score_change:.2f}")
        # col3.metric("Data Reliability", reliability_label)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
                <div style="padding:10px;border:1px solid #ddd;border-radius:8px;">
                    <div style="font-size:14px;color:gray;">Clinical Stability</div>
                    <div style="font-size:18px;font-weight:600;">{stability_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.metric("Clinical Score Change", f"{score_change:.2f}")

        with col3:
            st.markdown(
                f"""
                <div style="padding:10px;border:1px solid #ddd;border-radius:8px;">
                    <div style="font-size:14px;color:gray;">Data Reliability</div>
                    <div style="font-size:18px;font-weight:600;">{reliability_label}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.info(reliability_message)

else:
    st.warning("This subject does not have both sessions available for comparison.")



section5_title = (
    "5. Failure Cases and Unstable Outputs"
    if view_mode == "Researcher View"
    else "5. Cases Requiring Review"
)

st.header(section5_title)

failure_cases = outputs[
    (outputs["confidence"] < review_threshold) |
    (outputs["missing_modalities"] >= 2)
].copy()

failure_cases["review_reason"] = failure_cases.apply(
    lambda row: "Low confidence and high missingness"
    if row["confidence"] < review_threshold and row["missing_modalities"] >= 2
    else "Low confidence"
    if row["confidence"] < review_threshold
    else "High missingness",
    axis=1
)

#st.write("Potential cases requiring review:")

st.dataframe(
    failure_cases[
        [
            "subject_id",
            "session",
            "diagnosis",
            "behavior_score",
            "confidence",
            "missing_modalities",
            "review_reason"
        ]
    ],
    use_container_width=True
)


st.header("6. Validation & Evaluation Dashboard")

if view_mode == "Researcher View":
    st.markdown("""
    This view focuses on model behavior, embedding stability, sensitivity to missingness,
    and comparison across model versions.
    """)

    st.subheader("A. Session Stability")

    embedding_cols = embedding_df.columns.tolist()

    combined = outputs.copy()
    combined[embedding_cols] = embedding_df

    session_drift_rows = []

    for subject_id, group in combined.groupby("subject_id"):
        if set(group["session"]) == {1, 2}:
            s1 = group[group["session"] == 1][embedding_cols].values[0]
            s2 = group[group["session"] == 2][embedding_cols].values[0]

            drift = np.linalg.norm(s1 - s2)

            diagnosis = group["diagnosis"].iloc[0]
            score_change = abs(
                group[group["session"] == 1]["behavior_score"].values[0]
                - group[group["session"] == 2]["behavior_score"].values[0]
            )

            session_drift_rows.append({
                "subject_id": subject_id,
                "diagnosis": diagnosis,
                "session_drift": drift,
                "behavior_score_change": score_change
            })

    session_drift_df = pd.DataFrame(session_drift_rows)

    fig_drift = px.histogram(
        session_drift_df,
        x="session_drift",
        color="diagnosis",
        title="Session-to-Session Embedding Drift",
        labels={
            "session_drift": "Embedding Distance Between Session 1 and Session 2"
        }
    )

    st.plotly_chart(fig_drift, use_container_width=True)

    st.caption(
        "Higher drift means the same subject moved more in embedding space between sessions. "
        "This could reflect real clinical change, noisy data, or model instability."
    )


    st.subheader("B. Clinical Alignment")

    fig_score = px.scatter(
        outputs,
        x="pca_1",
        y="pca_2",
        color="behavior_score",
        hover_data=["subject_id", "session", "diagnosis", "confidence"],
        title="Embedding Space Colored by Clinical / Behavioral Score",
        labels={
            "pca_1": "PCA Component 1",
            "pca_2": "PCA Component 2",
            "behavior_score": "Clinical / Behavioral Score"
        }
    )

    st.plotly_chart(fig_score, use_container_width=True)


    st.subheader("C. Confidence vs Missingness")

    fig_conf_missing = px.box(
        outputs,
        x="missing_modalities",
        y="confidence",
        color="diagnosis",
        title="Model Confidence by Number of Missing Modalities",
        labels={
            "missing_modalities": "Number of Missing Modalities",
            "confidence": "Model Confidence"
        }
    )

    st.plotly_chart(fig_conf_missing, use_container_width=True)



    st.subheader("D. Model-Version Comparison")

    outputs_v1, embedding_df_v1, _ = generate_model_outputs(
        metadata,
        model_version="v1",
        embedding_dim=128,
        enabled_modalities=enabled_modalities,
        sensitivity_noise=sensitivity_noise
    )

    outputs_v2, embedding_df_v2, _ = generate_model_outputs(
        metadata,
        model_version="v2",
        embedding_dim=128,
        enabled_modalities=enabled_modalities,
        sensitivity_noise=sensitivity_noise
    )


    def compute_average_session_drift(outputs_df, embedding_df_local):
        temp = outputs_df.copy()
        temp[embedding_df_local.columns.tolist()] = embedding_df_local

        drift_values = []

        for subject_id, group in temp.groupby("subject_id"):
            if set(group["session"]) == {1, 2}:
                s1 = group[group["session"] == 1][embedding_df_local.columns.tolist()].values[0]
                s2 = group[group["session"] == 2][embedding_df_local.columns.tolist()].values[0]
                drift_values.append(np.linalg.norm(s1 - s2))

        return np.mean(drift_values)


    avg_drift_v1 = compute_average_session_drift(outputs_v1, embedding_df_v1)
    avg_drift_v2 = compute_average_session_drift(outputs_v2, embedding_df_v2)

    avg_conf_v1 = outputs_v1["confidence"].mean()
    avg_conf_v2 = outputs_v2["confidence"].mean()

    comparison_df = pd.DataFrame({
        "Metric": [
            "Average Session Drift",
            "Average Confidence",
            "Low Confidence Cases"
        ],
        "Model v1": [
            avg_drift_v1,
            avg_conf_v1,
            (outputs_v1["confidence"] < review_threshold).sum()
        ],
        "Model v2": [
            avg_drift_v2,
            avg_conf_v2,
            (outputs_v2["confidence"] < review_threshold).sum()
        ]
    })

    st.dataframe(comparison_df, use_container_width=True)

    comparison_long = comparison_df.melt(
        id_vars="Metric",
        var_name="Model Version",
        value_name="Value"
    )

    fig_model_compare = px.bar(
        comparison_long,
        x="Metric",
        y="Value",
        color="Model Version",
        barmode="group",
        title="Model v1 vs v2 Evaluation Summary"
    )

    st.plotly_chart(fig_model_compare, use_container_width=True)

    st.caption(
        "This comparison supports iterative model development. A newer model may be preferred if it improves stability, "
        "confidence, or robustness, even before a final gold-standard metric is agreed upon."
    )



else:
    st.markdown("""
    This view summarizes model reliability in plain language for clinical and research staff.
    It does not require interpretation of embedding geometry.
    """)

    st.subheader("A. Clinical Reliability Summary")

    total_cases = len(outputs)
    caution_cases = len(
        outputs[
            (outputs["confidence"] < review_threshold) |
            (outputs["missing_modalities"] >= 2)
        ]
    )

    low_confidence_cases = len(outputs[outputs["confidence"] < review_threshold])
    high_missing_cases = len(outputs[outputs["missing_modalities"] >= 2])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Subject-Sessions", total_cases)
    col2.metric("Use-Caution Cases", caution_cases)
    col3.metric("Low-Confidence Cases", low_confidence_cases)

    st.info(
        f"{caution_cases} out of {total_cases} subject-sessions are flagged for cautious interpretation "
        f"based on low confidence or substantial missing data."
    )

    st.subheader("B. Data Completeness Summary")

    completeness = outputs["missing_modalities"].value_counts().sort_index().reset_index()
    completeness.columns = ["Missing Modalities", "Subject-Sessions"]

    fig_completeness = px.bar(
        completeness,
        x="Missing Modalities",
        y="Subject-Sessions",
        title="Data Completeness Across Subject-Sessions"
    )

    st.plotly_chart(fig_completeness, use_container_width=True)

    st.subheader("C. Reliability by Diagnosis Group")

    reliability_df = outputs.copy()
    reliability_df["reliability_status"] = reliability_df.apply(
        lambda row: "Use caution"
        if row["confidence"] < review_threshold or row["missing_modalities"] >= 2
        else "Likely reliable",
        axis=1
    )

    reliability_summary = reliability_df.groupby(
        ["diagnosis", "reliability_status"]
    ).size().reset_index(name="count")

    fig_reliability = px.bar(
        reliability_summary,
        x="diagnosis",
        y="count",
        color="reliability_status",
        barmode="group",
        title="Reliability Summary by Diagnosis Group"
    )

    st.plotly_chart(fig_reliability, use_container_width=True)

    st.subheader("D. Plain-Language Model Comparison")

    st.markdown("""
    The current model version is compared using practical reliability signals rather than a single accuracy score.
    A preferred model should generally produce fewer caution cases, maintain reasonable confidence,
    and remain robust when some data are missing.
    """)

    outputs_v1, embedding_df_v1, _ = generate_model_outputs(
        metadata,
        model_version="v1",
        embedding_dim=128,
        enabled_modalities=enabled_modalities,
        sensitivity_noise=sensitivity_noise
    )

    outputs_v2, embedding_df_v2, _ = generate_model_outputs(
        metadata,
        model_version="v2",
        embedding_dim=128,
        enabled_modalities=enabled_modalities,
        sensitivity_noise=sensitivity_noise
    )

    v1_caution = len(
        outputs_v1[
            (outputs_v1["confidence"] < review_threshold) |
            (outputs_v1["missing_modalities"] >= 2)
        ]
    )

    v2_caution = len(
        outputs_v2[
            (outputs_v2["confidence"] < review_threshold) |
            (outputs_v2["missing_modalities"] >= 2)
        ]
    )

    model_summary = pd.DataFrame({
        "Model": ["v1 CNN-based encoder", "v2 Transformer-based encoder"],
        "Use-caution cases": [v1_caution, v2_caution],
        "Average confidence": [
            outputs_v1["confidence"].mean(),
            outputs_v2["confidence"].mean()
        ]
    })

    st.dataframe(model_summary, use_container_width=True)

    if v2_caution < v1_caution:
        st.success(
            "In this simulated run, v2 has fewer caution cases and may be easier to interpret clinically."
        )
    elif v2_caution > v1_caution:
        st.warning(
            "In this simulated run, v2 has more caution cases. This would require further investigation."
        )
    else:
        st.info(
            "Both model versions produce a similar number of caution cases under the current settings."
        )


# st.header("6. Validation & Evaluation Dashboard")

# st.markdown("""
# Because this is an early-stage biomedical ML model, evaluation is not reduced to one accuracy number.
# Instead, this prototype uses multiple validation signals: stability, missingness sensitivity,
# clinical alignment, uncertainty, and model-version comparison.
# """)

# st.subheader("A. Session Stability")

# embedding_cols = embedding_df.columns.tolist()

# combined = outputs.copy()
# combined[embedding_cols] = embedding_df

# session_drift_rows = []

# for subject_id, group in combined.groupby("subject_id"):
#     if set(group["session"]) == {1, 2}:
#         s1 = group[group["session"] == 1][embedding_cols].values[0]
#         s2 = group[group["session"] == 2][embedding_cols].values[0]

#         drift = np.linalg.norm(s1 - s2)

#         diagnosis = group["diagnosis"].iloc[0]
#         score_change = abs(
#             group[group["session"] == 1]["behavior_score"].values[0]
#             - group[group["session"] == 2]["behavior_score"].values[0]
#         )

#         session_drift_rows.append({
#             "subject_id": subject_id,
#             "diagnosis": diagnosis,
#             "session_drift": drift,
#             "behavior_score_change": score_change
#         })

# session_drift_df = pd.DataFrame(session_drift_rows)

# fig_drift = px.histogram(
#     session_drift_df,
#     x="session_drift",
#     color="diagnosis",
#     title="Session-to-Session Embedding Drift",
#     labels={
#         "session_drift": "Embedding Distance Between Session 1 and Session 2"
#     }
# )

# st.plotly_chart(fig_drift, use_container_width=True)

# st.caption(
#     "Higher drift means the same subject moved more in embedding space between sessions. "
#     "This could reflect real clinical change, noisy data, or model instability."
# )


# st.subheader("B. Clinical Alignment")

# fig_score = px.scatter(
#     outputs,
#     x="pca_1",
#     y="pca_2",
#     color="behavior_score",
#     hover_data=["subject_id", "session", "diagnosis", "confidence"],
#     title="Embedding Space Colored by Clinical / Behavioral Score",
#     labels={
#         "pca_1": "PCA Component 1",
#         "pca_2": "PCA Component 2",
#         "behavior_score": "Clinical / Behavioral Score"
#     }
# )

# st.plotly_chart(fig_score, use_container_width=True)


# st.subheader("C. Confidence vs Missingness")

# fig_conf_missing = px.box(
#     outputs,
#     x="missing_modalities",
#     y="confidence",
#     color="diagnosis",
#     title="Model Confidence by Number of Missing Modalities",
#     labels={
#         "missing_modalities": "Number of Missing Modalities",
#         "confidence": "Model Confidence"
#     }
# )

# st.plotly_chart(fig_conf_missing, use_container_width=True)



# st.subheader("D. Model-Version Comparison")

# outputs_v1, embedding_df_v1, _ = generate_model_outputs(
#     metadata,
#     model_version="v1",
#     embedding_dim=128,
#     enabled_modalities=enabled_modalities,
#     sensitivity_noise=sensitivity_noise
# )

# outputs_v2, embedding_df_v2, _ = generate_model_outputs(
#     metadata,
#     model_version="v2",
#     embedding_dim=128,
#     enabled_modalities=enabled_modalities,
#     sensitivity_noise=sensitivity_noise
# )


# def compute_average_session_drift(outputs_df, embedding_df_local):
#     temp = outputs_df.copy()
#     temp[embedding_df_local.columns.tolist()] = embedding_df_local

#     drift_values = []

#     for subject_id, group in temp.groupby("subject_id"):
#         if set(group["session"]) == {1, 2}:
#             s1 = group[group["session"] == 1][embedding_df_local.columns.tolist()].values[0]
#             s2 = group[group["session"] == 2][embedding_df_local.columns.tolist()].values[0]
#             drift_values.append(np.linalg.norm(s1 - s2))

#     return np.mean(drift_values)


# avg_drift_v1 = compute_average_session_drift(outputs_v1, embedding_df_v1)
# avg_drift_v2 = compute_average_session_drift(outputs_v2, embedding_df_v2)

# avg_conf_v1 = outputs_v1["confidence"].mean()
# avg_conf_v2 = outputs_v2["confidence"].mean()

# comparison_df = pd.DataFrame({
#     "Metric": [
#         "Average Session Drift",
#         "Average Confidence",
#         "Low Confidence Cases"
#     ],
#     "Model v1": [
#         avg_drift_v1,
#         avg_conf_v1,
#         (outputs_v1["confidence"] < review_threshold).sum()
#     ],
#     "Model v2": [
#         avg_drift_v2,
#         avg_conf_v2,
#         (outputs_v2["confidence"] < review_threshold).sum()
#     ]
# })

# st.dataframe(comparison_df, use_container_width=True)

# comparison_long = comparison_df.melt(
#     id_vars="Metric",
#     var_name="Model Version",
#     value_name="Value"
# )

# fig_model_compare = px.bar(
#     comparison_long,
#     x="Metric",
#     y="Value",
#     color="Model Version",
#     barmode="group",
#     title="Model v1 vs v2 Evaluation Summary"
# )

# st.plotly_chart(fig_model_compare, use_container_width=True)

# st.caption(
#     "This comparison supports iterative model development. A newer model may be preferred if it improves stability, "
#     "confidence, or robustness, even before a final gold-standard metric is agreed upon."
# )


# import streamlit as st
# import pandas as pd
# import plotly.express as px

# from synthetic_data import generate_subject_metadata, generate_model_outputs


# st.set_page_config(
#     page_title="Biomedical Model Explorer",
#     layout="wide"
# )

# st.title("Biomedical Model Explorer")
# st.caption("Prototype for exploring evolving multimodal biomedical ML models")


# metadata = generate_subject_metadata()

# st.sidebar.header("Controls")

# model_version = st.sidebar.selectbox(
#     "Model version",
#     ["v1", "v2"]
# )

# view_mode = st.sidebar.radio(
#     "User view",
#     ["Researcher View", "Clinician View"]
# )

# outputs, embedding_df, pca = generate_model_outputs(
#     metadata,
#     model_version=model_version,
#     embedding_dim=128
# )

