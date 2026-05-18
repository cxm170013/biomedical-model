import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def generate_subject_metadata(n_subjects=30, seed=42):
    np.random.seed(seed)

    subjects = []
    diagnoses = ["Control", "Mild Disorder", "Severe Disorder"]

    for i in range(1, n_subjects + 1):
        subject_id = f"S{i:03d}"
        diagnosis = np.random.choice(diagnoses, p=[0.35, 0.4, 0.25])

        subject_baseline = np.random.normal(0, 0.4)

        for session in [1, 2]:
            subjects.append({
                "subject_id": subject_id,
                "session": session,
                "diagnosis": diagnosis,
                "behavior_score": np.random.normal(
                    loc={"Control": 25, "Mild Disorder": 50, "Severe Disorder": 75}[diagnosis],
                    scale=8
                ),
                "speech_available": np.random.rand() > 0.10,
                "wearable_available": np.random.rand() > 0.20,
                "mri_available": np.random.rand() > 0.30,
                "subject_baseline": subject_baseline
            })

    return pd.DataFrame(subjects)


def generate_model_outputs(
    metadata,
    model_version="v1",
    embedding_dim=128,
    enabled_modalities=None,
    sensitivity_noise=1.0,
    seed=42
):
    if enabled_modalities is None:
        enabled_modalities = ["speech", "wearable", "mri"]

    np.random.seed(seed if model_version == "v1" else seed + 10)

    modality_weights = {
        "speech": 0.30,
        "wearable": 0.30,
        "mri": 0.40
    }

    active_weight = sum(modality_weights[m] for m in enabled_modalities)

    embeddings = []
    rows = []

    for _, row in metadata.iterrows():
        diagnosis_shift = {
            "Control": 0.0,
            "Mild Disorder": 1.5,
            "Severe Disorder": 3.0
        }[row["diagnosis"]]

        disease_drift_noise = {
            "Control": 0.25,
            "Mild Disorder": 0.45,
            "Severe Disorder": 0.70
        }[row["diagnosis"]]

        model_noise = 1.2 if model_version == "v1" else 0.9

        session_effect = np.random.normal(
            0,
            disease_drift_noise * sensitivity_noise,
            embedding_dim
        )

        modality_effect = active_weight * diagnosis_shift

        embedding = np.random.normal(
            loc=modality_effect + row["subject_baseline"],
            scale=model_noise * sensitivity_noise,
            size=embedding_dim
        ) + session_effect

        actual_available = {
            "speech": row["speech_available"],
            "wearable": row["wearable_available"],
            "mri": row["mri_available"]
        }

        unavailable_due_to_data = sum(
            not actual_available[m] for m in ["speech", "wearable", "mri"]
        )

        disabled_by_user = 3 - len(enabled_modalities)

        #missing_count = unavailable_due_to_data + disabled_by_user

        missing_count = min(unavailable_due_to_data + disabled_by_user,2)

        confidence = max(
            0.1,
            min(
                0.95,
                np.random.normal(
                    0.85 - missing_count * 0.12 - (sensitivity_noise - 1.0) * 0.10,
                    0.08
                )
            )
        )

        embeddings.append(embedding)

        rows.append({
            "subject_id": row["subject_id"],
            "session": row["session"],
            "diagnosis": row["diagnosis"],
            "behavior_score": row["behavior_score"],
            "confidence": confidence,
            "missing_modalities": missing_count,
            "model_version": model_version,
            "active_modalities": ", ".join(enabled_modalities)
        })

    outputs = pd.DataFrame(rows)

    embedding_columns = [f"embedding_{i+1}" for i in range(embedding_dim)]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)

    pca = PCA(n_components=2)
    projection = pca.fit_transform(embedding_df)

    outputs["pca_1"] = projection[:, 0]
    outputs["pca_2"] = projection[:, 1]

    return outputs, embedding_df, pca

# import numpy as np
# import pandas as pd
# from sklearn.decomposition import PCA


# def generate_subject_metadata(n_subjects=30, seed=42):
#     np.random.seed(seed)

#     subjects = []
#     diagnoses = ["Control", "Mild Disorder", "Severe Disorder"]

#     for i in range(1, n_subjects + 1):
#         subject_id = f"S{i:03d}"
#         diagnosis = np.random.choice(diagnoses, p=[0.35, 0.4, 0.25])

#         for session in [1, 2]:
#             subjects.append({
#                 "subject_id": subject_id,
#                 "session": session,
#                 "diagnosis": diagnosis,
#                 "behavior_score": np.random.normal(
#                     loc={"Control": 25, "Mild Disorder": 50, "Severe Disorder": 75}[diagnosis],
#                     scale=8
#                 ),
#                 "speech_available": np.random.rand() > 0.10,
#                 "wearable_available": np.random.rand() > 0.20,
#                 "mri_available": np.random.rand() > 0.30,
#             })

#     return pd.DataFrame(subjects)


# def generate_model_outputs(metadata, model_version="v1", embedding_dim=128, seed=42):
#     np.random.seed(seed if model_version == "v1" else seed + 10)

#     embeddings = []
#     rows = []

#     for _, row in metadata.iterrows():
#         diagnosis_shift = {
#             "Control": 0.0,
#             "Mild Disorder": 1.5,
#             "Severe Disorder": 3.0
#         }[row["diagnosis"]]

#         noise = 1.2 if model_version == "v1" else 0.9

#         # Simulated 128D model embedding
#         embedding = np.random.normal(
#             loc=diagnosis_shift,
#             scale=noise,
#             size=embedding_dim
#         )

#         missing_count = 3 - sum([
#             row["speech_available"],
#             row["wearable_available"],
#             row["mri_available"]
#         ])

#         confidence = max(
#             0.1,
#             min(0.95, np.random.normal(0.8 - missing_count * 0.15, 0.08))
#         )

#         embeddings.append(embedding)

#         rows.append({
#             "subject_id": row["subject_id"],
#             "session": row["session"],
#             "diagnosis": row["diagnosis"],
#             "behavior_score": row["behavior_score"],
#             "confidence": confidence,
#             "missing_modalities": missing_count,
#             "model_version": model_version
#         })

#     outputs = pd.DataFrame(rows)

#     embedding_columns = [f"embedding_{i+1}" for i in range(embedding_dim)]
#     embedding_df = pd.DataFrame(embeddings, columns=embedding_columns)

#     pca = PCA(n_components=2)
#     projection = pca.fit_transform(embedding_df)

#     outputs["pca_1"] = projection[:, 0]
#     outputs["pca_2"] = projection[:, 1]

#     return outputs, embedding_df, pca
