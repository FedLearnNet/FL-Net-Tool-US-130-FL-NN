"""
Preprocessing, output builders, and HTML report generation for the
Hospital Readmission Prediction app (test_us130_wide schema).
"""

import json

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGE_MAP = {
    "[0-10)": 5,
    "[10-20)": 15,
    "[20-30)": 25,
    "[30-40)": 35,
    "[40-50)": 45,
    "[50-60)": 55,
    "[60-70)": 65,
    "[70-80)": 75,
    "[80-90)": 85,
    "[90-100)": 95,
}

# All medication columns in the wide schema (No / Steady / Up / Down → 0/1)
MEDICATION_COLS = [
    "metformin_pioglitazone", "metformin_rosiglitazone", "glimepiride_pioglitazone",
    "glipizide_metformin", "glyburide_metformin", "insulin", "citoglipton",
    "examide", "tolazamide", "troglitazone", "miglitol", "acarbose",
    "rosiglitazone", "pioglitazone", "tolbutamide", "glyburide", "glipizide",
    "acetohexamide", "glimepiride", "chlorpropamide", "nateglinide",
    "repaglinide", "metformin",
]

# Columns to discard before modelling
DROP_COLS = [
    # IDs — pure identifiers, never model features. One-hot of these is noise/leakage and, once
    # cohorts are pooled, would explode into one dummy column per encounter.
    "patient_id", "patient_number", "encounter_id", "encounter_identifier",
    # Redundant age representations
    "age", "age_age_years",
    # Entirely null in this dataset
    "height", "body_height_observable_entity",
    "diastolic_blood_pressure_observable_entity",
    "systolic_blood_pressure_observable_entity_diastolic_blood_pressure_observable_entity",
    "diabetes", "diabetes_mellitus_type_2_disorder",
    # Mostly null (~96%)
    "body_weight", "weight", "body_weight_observable_entity",
    # Low signal / mostly null
    "payer_code",
    # Gender duplicates
    "gender_observable_entity", "sex_assigned_at_birth",
]

# Categorical columns to be cast to str before one-hot encoding
# (admission codes come as ints in this dataset)
CAST_TO_STR_COLS = ["admission_source", "discharge_disposition", "admission_type"]

DIAG_COLS = ["primary_diagnosis", "secondary_diagnosis", "tertiary_diagnosis"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_diagnosis(code) -> str:
    """Map an ICD-9 code string to a broad clinical category."""
    try:
        code = str(code).strip()
        if not code or code in ("nan", "?"):
            return "Other"
        if code.startswith("V") or code.startswith("E"):
            return "Other"
        num = float(code.split(".")[0])
        if 390 <= num <= 459 or num == 785:
            return "Circulatory"
        if 460 <= num <= 519 or num == 786:
            return "Respiratory"
        if 520 <= num <= 579 or num == 787:
            return "Digestive"
        if 250 <= num < 251:
            return "Diabetes"
        if 800 <= num <= 999:
            return "Injury"
        if 710 <= num <= 739:
            return "Musculoskeletal"
        if 580 <= num <= 629 or num == 788:
            return "Genitourinary"
        if 140 <= num <= 239:
            return "Neoplasms"
        return "Other"
    except (ValueError, AttributeError):
        return "Other"


# ---------------------------------------------------------------------------
# Public preprocessing function
# ---------------------------------------------------------------------------

def preprocess_diabetic(
    df: pd.DataFrame,
    target_col: str = "hospital_readmission",
    is_training: bool = True,
    feature_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[str]]:
    """
    Full preprocessing pipeline for the test_us130_wide schema.

    Returns
    -------
    X            : float64 ndarray (n_samples, n_features)
    y            : int ndarray (n_samples,) — 1 if readmitted <30d, else 0; None in predict mode
    feature_cols : list[str] matching X columns
    """
    df = df.copy()

    # Replace sentinel missing values
    df.replace("?", np.nan, inplace=True)

    # Drop irrelevant columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Map age bracket to numeric midpoint
    if "age_age_group_bracket" in df.columns:
        df["age_numeric"] = df["age_age_group_bracket"].map(AGE_MAP)
        df.drop(columns=["age_age_group_bracket"], inplace=True)

    # Map ICD-9 diagnosis codes to clinical categories
    for diag in DIAG_COLS:
        if diag in df.columns:
            df[f"{diag}_cat"] = df[diag].apply(_map_diagnosis)
            df.drop(columns=[diag], inplace=True)

    # Binarize medication columns: No → 0, any prescription / dose change → 1
    for med in MEDICATION_COLS:
        if med in df.columns:
            df[med] = (df[med].fillna("No") != "No").astype(np.int8)

    # Binarize Yes/No flags
    if "diabetes_medication_prescribed" in df.columns:
        df["diabetes_medication_prescribed"] = (
            df["diabetes_medication_prescribed"].fillna("No") == "Yes"
        ).astype(np.int8)

    if "medication_change" in df.columns:
        df["medication_change"] = (
            df["medication_change"].fillna("No") == "Ch"
        ).astype(np.int8)

    # Cast numeric-coded categoricals to string so they get one-hot encoded
    for col in CAST_TO_STR_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Fill mostly-missing semi-useful columns with a dedicated category
    for col in ["maximum_glucose_serum", "hb_a1c"]:
        if col in df.columns:
            df[col] = df[col].fillna("None")

    # Extract and binarize target: "<30" → 1 (readmitted within 30 days), else 0
    y = None
    if target_col in df.columns:
        raw = df.pop(target_col)
        y = (raw == "<30").astype(int).values

    # Drop columns that are now entirely NaN
    df.dropna(axis=1, how="all", inplace=True)

    # One-hot encode all remaining string/object columns
    cat_cols = [c for c in df.columns if df[c].dtype == object]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    # Impute remaining NaN with column median
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Prediction mode: align feature columns to training schema
    if not is_training and feature_columns is not None:
        df = df.reindex(columns=feature_columns, fill_value=0)

    return df.values.astype(np.float64), y, list(df.columns)


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def build_prediction_df(
    ids: list,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ids,
            "readmitted_30d_predicted": y_pred.astype(int),
            "probability_readmitted": np.round(y_proba, 4),
            "probability_not_readmitted": np.round(1.0 - y_proba, 4),
        }
    )


def coef_table(
    feature_names: list[str],
    coef: np.ndarray,
    intercept: np.ndarray,
) -> pd.DataFrame:
    rows = [
        {"feature": name, "coefficient": float(c)}
        for name, c in zip(feature_names, coef[0])
    ]
    rows.append({"feature": "__intercept__", "coefficient": float(intercept[0])})
    return (
        pd.DataFrame(rows)
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def generate_explanation_html(
    coef_df: pd.DataFrame,
    metrics: dict,
    confusion_mat: list[list[int]],
    classification_rep: str,
    training_history: list[dict] | None = None,
    top_k: int = 20,
    val_fpr: np.ndarray | None = None,
    val_tpr: np.ndarray | None = None,
) -> str:
    """Generate an interactive HTML report with ECharts visualizations."""

    features_only = coef_df[coef_df["feature"] != "__intercept__"]
    top_pos = features_only.nlargest(top_k, "coefficient")
    top_neg = features_only.nsmallest(top_k, "coefficient")
    top_features = (
        pd.concat([top_pos, top_neg])
        .drop_duplicates("feature")
        .sort_values("coefficient")
    )

    metric_rows = "\n".join(
        f"  <tr><td>{k.replace('_', ' ').title()}</td>"
        f"<td><b>{v:.4f}</b></td></tr>"
        if isinstance(v, float)
        else f"  <tr><td>{k.replace('_', ' ').title()}</td><td><b>{v}</b></td></tr>"
        for k, v in metrics.items()
    )

    tn = confusion_mat[0][0]
    fp = confusion_mat[0][1]
    fn = confusion_mat[1][0]
    tp = confusion_mat[1][1]

    bar_features_json = json.dumps(top_features["feature"].tolist())
    bar_vals = [round(float(v), 4) for v in top_features["coefficient"]]
    bar_values_json = json.dumps(bar_vals)
    bar_colors_json = json.dumps(
        ["#c0392b" if v < 0 else "#27ae60" for v in bar_vals]
    )

    # Training history chart (accuracy + AUC over iterations)
    history_html = ""
    if training_history:
        x_vals      = json.dumps([h["x"] for h in training_history])
        train_acc   = json.dumps([round(h["train_accuracy"], 4) for h in training_history])
        val_acc     = json.dumps([round(h["val_accuracy"],   4) for h in training_history])
        train_auc_h = json.dumps([round(h["train_auc"],      4) for h in training_history])
        val_auc_h   = json.dumps([round(h["val_auc"],        4) for h in training_history])
        history_html = f"""
<h2>Training History</h2>
<div id="history_chart" style="width:100%;height:380px;"></div>
<script>
(function() {{
  var c = echarts.init(document.getElementById('history_chart'));
  var xs = {x_vals};
  c.setOption({{
    title: {{ text: 'Metrics over Iterations', left: 'center' }},
    tooltip: {{ trigger: 'axis' }},
    legend: {{ bottom: 0 }},
    xAxis: {{ type: 'category', data: xs, name: 'Iterations', nameLocation: 'center', nameGap: 28 }},
    yAxis: {{ type: 'value', min: 0, max: 1 }},
    series: [
      {{ name: 'Train Accuracy', type: 'line', data: {train_acc}, smooth: true, lineStyle: {{ color: '#3498db' }} }},
      {{ name: 'Val Accuracy',   type: 'line', data: {val_acc},   smooth: true, lineStyle: {{ color: '#2980b9', type: 'dashed' }} }},
      {{ name: 'Train AUC',      type: 'line', data: {train_auc_h}, smooth: true, lineStyle: {{ color: '#e67e22' }} }},
      {{ name: 'Val AUC',        type: 'line', data: {val_auc_h},   smooth: true, lineStyle: {{ color: '#d35400', type: 'dashed' }} }}
    ]
  }});
}})();
</script>"""

    # ROC curve
    roc_html = ""
    if val_fpr is not None and val_tpr is not None:
        step = max(1, len(val_fpr) // 300)
        roc_data = json.dumps(
            [[round(float(x), 4), round(float(y), 4)]
             for x, y in zip(val_fpr[::step], val_tpr[::step])]
        )
        roc_auc = metrics.get("val_auc", 0.0)
        roc_html = f"""
<h2>ROC Curve (Validation Set)</h2>
<div id="roc_chart" style="width:100%;height:400px;"></div>
<script>
(function() {{
  var c = echarts.init(document.getElementById('roc_chart'));
  c.setOption({{
    title: {{ text: 'ROC Curve — AUC = {roc_auc:.4f}', left: 'center' }},
    tooltip: {{ trigger: 'axis' }},
    xAxis: {{ type: 'value', name: 'False Positive Rate', min: 0, max: 1,
              nameLocation: 'center', nameGap: 28 }},
    yAxis: {{ type: 'value', name: 'True Positive Rate', min: 0, max: 1,
              nameLocation: 'center', nameGap: 36 }},
    series: [
      {{ name: 'ROC', type: 'line', data: {roc_data},
         showSymbol: false, lineStyle: {{ color: '#2980b9', width: 2 }} }},
      {{ name: 'Random', type: 'line', data: [[0,0],[1,1]],
         showSymbol: false, lineStyle: {{ color: '#95a5a6', type: 'dashed' }} }}
    ]
  }});
}})();
</script>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hospital Readmission — Model Report</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 0 auto; padding: 24px;
         color: #2c3e50; line-height: 1.5; }}
  h1 {{ border-bottom: 2px solid #2c3e50; padding-bottom: 8px; }}
  h2 {{ color: #2980b9; margin-top: 32px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th {{ background: #2c3e50; color: #fff; padding: 8px 14px; text-align: left; }}
  td {{ padding: 6px 14px; border-bottom: 1px solid #ecf0f1; }}
  tr:nth-child(even) td {{ background: #f8f9fa; }}
  .cm td {{ text-align: center; padding: 12px 24px; font-size: 1.1em; }}
  .good {{ background: #d5f5e3; }}
  .bad  {{ background: #fadbd8; }}
  pre  {{ background: #f8f9fa; padding: 16px; border-left: 4px solid #2980b9;
          overflow-x: auto; font-size: 13px; }}
</style>
</head>
<body>
<h1>Hospital Readmission Prediction — Model Report</h1>
<p>Neural network predicting <strong>30-day readmission</strong> for diabetic patients.</p>

<h2>Training &amp; Validation Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
{metric_rows}
</table>

<h2>Confusion Matrix (Validation Set)</h2>
<table class="cm">
  <tr>
    <th></th>
    <th>Predicted: Not Readmitted</th>
    <th>Predicted: Readmitted (&lt;30d)</th>
  </tr>
  <tr>
    <td><b>Actual: Not Readmitted</b></td>
    <td class="good">TN&nbsp;=&nbsp;{tn}</td>
    <td class="bad">FP&nbsp;=&nbsp;{fp}</td>
  </tr>
  <tr>
    <td><b>Actual: Readmitted (&lt;30d)</b></td>
    <td class="bad">FN&nbsp;=&nbsp;{fn}</td>
    <td class="good">TP&nbsp;=&nbsp;{tp}</td>
  </tr>
</table>

<h2>Classification Report</h2>
<pre>{classification_rep}</pre>

{history_html}

{roc_html}

<h2>Feature Importance — Top {top_k} Positive &amp; Negative Input Weights</h2>
<div id="coef_chart" style="width:100%;height:620px;"></div>
<script>
(function() {{
  var c = echarts.init(document.getElementById('coef_chart'));
  var features = {bar_features_json};
  var values   = {bar_values_json};
  var colors   = {bar_colors_json};
  c.setOption({{
    title: {{ text: 'First-Layer Feature Weights (Neural Network)', left: 'center' }},
    tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
    grid: {{ left: '38%', right: '12%', containLabel: false }},
    xAxis: {{ type: 'value', name: 'Mean first-layer weight', nameLocation: 'center', nameGap: 26 }},
    yAxis: {{ type: 'category', data: features, axisLabel: {{ fontSize: 11 }} }},
    series: [{{
      name: 'Mean first-layer weight',
      type: 'bar',
      data: values.map(function(v, i) {{
        return {{ value: v, itemStyle: {{ color: colors[i] }} }};
      }}),
      label: {{
        show: true, position: 'right',
        formatter: function(p) {{ return p.value.toFixed(3); }}
      }}
    }}]
  }});
}})();
</script>

</body>
</html>"""
    return html
