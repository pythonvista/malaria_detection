#!/usr/bin/env python3
"""
Generate professional system diagrams using Graphviz.

The diagrams mirror the Mermaid definitions in MERMAID_DIAGRAM_CODES.md but are rendered
locally via the Graphviz Python bindings for higher-quality visual output.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from graphviz import Digraph

DiagramSpec = Dict[str, object]

THEME = {
    "font": "Helvetica",
    "font_bold": "Helvetica-Bold",
    "background": "#f8fafc",
    "cluster_label_color": "#0f172a",
    "cluster_border": "#cbd5f5",
    "node": {
        "shape": "rect",
        "style": "rounded,filled,setlinewidth(2)",
        "color": "#0f172a",
        "fillcolor": "#ffffff",
        "fontcolor": "#0f172a",
        "fontsize": "11",
    },
    "edge": {
        "color": "#475569",
        "penwidth": "1.6",
        "arrowsize": "0.9",
        "arrowhead": "vee",
        "fontname": "Helvetica",
    },
}

OUTPUT_FORMATS = ("png", "svg")
GRAPHVIZ_BIN_CANDIDATES = [
    Path(os.environ.get("GRAPHVIZ_BIN", "")),
    Path("C:/Program Files/Graphviz/bin"),
    Path("C:/Program Files (x86)/Graphviz/bin"),
]


def ensure_graphviz_binary() -> None:
    """Ensure Graphviz executables (dot) are available on PATH."""
    if shutil.which("dot"):
        return

    for candidate in GRAPHVIZ_BIN_CANDIDATES:
        if not candidate:
            continue
        dot_path = candidate / "dot.exe"
        if dot_path.exists():
            os.environ["PATH"] = f"{candidate}{os.pathsep}{os.environ['PATH']}"
            return

    raise RuntimeError(
        "Graphviz 'dot' executable not found. Install Graphviz or set GRAPHVIZ_BIN to the bin directory."
    )

DIAGRAMS: Dict[str, DiagramSpec] = {
    "system_architecture": {
        "rankdir": "TB",
        "clusters": [
            {
                "name": "cluster_input",
                "label": "DATA INPUT LAYER",
                "fill": "#e1f5ff",
                "nodes": [
                    ("A", "Patient Information\nAge, Gender, Location"),
                    ("B", "Clinical Symptoms\n15 Malaria Symptoms"),
                ],
            },
            {
                "name": "cluster_preproc",
                "label": "PREPROCESSING UNIT",
                "fill": "#fff4e1",
                "nodes": [
                    ("C", "Data Validation"),
                    ("D", "Data Cleaning"),
                    ("E", "Feature Encoding"),
                    ("F", "Normalization"),
                ],
            },
            {
                "name": "cluster_analysis",
                "label": "AI ANALYSIS ENGINE",
                "fill": "#f0e1ff",
                "nodes": [
                    ("G", "Feature Extraction"),
                    ("H", "Pattern Recognition"),
                    ("I", "ML Model Selection"),
                ],
            },
            {
                "name": "cluster_models",
                "label": "MACHINE LEARNING MODELS",
                "fill": "#e1ffe1",
                "nodes": [
                    ("J", "Decision Tree\n85.4% Accuracy"),
                    ("K", "SVM Model\n88.2% Accuracy"),
                    ("L", "Logistic Regression\n86.8% Accuracy"),
                ],
            },
            {
                "name": "cluster_output",
                "label": "OUTPUT INTERFACE",
                "fill": "#ffe1e1",
                "nodes": [
                    ("M", "Prediction Result\nPositive / Negative"),
                    ("N", "Confidence Score"),
                    ("O", "Clinical Recommendations"),
                ],
            },
            {
                "name": "cluster_report",
                "label": "DIAGNOSIS REPORT",
                "fill": "#ffe1f5",
                "nodes": [
                    ("P", "Final Report Generation"),
                    ("Q", "Treatment Suggestions"),
                    ("R", "Follow-up Actions"),
                ],
            },
        ],
        "edges": [
            ("A", "C"),
            ("B", "C"),
            ("C", "D"),
            ("D", "E"),
            ("E", "F"),
            ("F", "G"),
            ("G", "H"),
            ("H", "I"),
            ("I", "J"),
            ("I", "K"),
            ("I", "L"),
            ("J", "M"),
            ("K", "M"),
            ("L", "M"),
            ("M", "N"),
            ("N", "O"),
            ("O", "P"),
            ("P", "Q"),
            ("Q", "R"),
        ],
    },
    "data_preprocessing": {
        "rankdir": "LR",
        "clusters": [
            {
                "name": "cluster_raw",
                "label": "RAW DATA INPUT",
                "fill": "#ffcccc",
                "nodes": [
                    ("A1", "Dataset: 2000 Records"),
                    ("A2", "17 Features\nAge, Gender, Symptoms"),
                    ("A3", "Target Variable\nSevere Malaria 0/1"),
                ],
            },
            {
                "name": "cluster_validation",
                "label": "DATA VALIDATION",
                "fill": "#ffffcc",
                "nodes": [
                    ("B1", "Check Missing Values"),
                    ("B2", "Check Data Types"),
                    ("B3", "Check Duplicates"),
                    ("B4", "Verify Data Integrity"),
                ],
            },
            {
                "name": "cluster_cleaning",
                "label": "DATA CLEANING",
                "fill": "#ccffcc",
                "nodes": [
                    ("C1", "Handle Missing Values\nMedian / Mode"),
                    ("C2", "Remove Duplicates"),
                    ("C3", "Fix Inconsistencies"),
                    ("C4", "Outlier Detection"),
                ],
            },
            {
                "name": "cluster_transform",
                "label": "DATA TRANSFORMATION",
                "fill": "#ccccff",
                "nodes": [
                    ("D1", "Label Encoding\nGender"),
                    ("D2", "Binary Encoding\nSymptoms"),
                    ("D3", "Feature Scaling\nStandardScaler"),
                    ("D4", "Normalization\nZ-score Method"),
                ],
            },
            {
                "name": "cluster_output",
                "label": "CLEAN DATA OUTPUT",
                "fill": "#ffccff",
                "nodes": [
                    ("E1", "Processed Dataset\n2000 Clean Records"),
                    ("E2", "Ready for Model Training"),
                ],
            },
        ],
        "edges": [
            ("A1", "B1"),
            ("A2", "B2"),
            ("A3", "B3"),
            ("B1", "B4"),
            ("B2", "B4"),
            ("B3", "B4"),
            ("B4", "C1"),
            ("C1", "C2"),
            ("C2", "C3"),
            ("C3", "C4"),
            ("C4", "D1"),
            ("D1", "D2"),
            ("D2", "D3"),
            ("D3", "D4"),
            ("D4", "E1"),
            ("E1", "E2"),
        ],
    },
    "feature_engineering": {
        "rankdir": "TB",
        "clusters": [
            {
                "name": "cluster_input",
                "label": "INPUT FEATURES (17 total)",
                "fill": "#e3f2fd",
                "nodes": [
                    ("A1", "Demographics\nAge 3-77, Gender M/F"),
                    ("A2", "Clinical Symptoms\n15 Binary Features"),
                ],
            },
            {
                "name": "cluster_analysis",
                "label": "FEATURE ANALYSIS",
                "fill": "#fff9c4",
                "nodes": [
                    ("B1", "Statistical Analysis\nMean, Median, STD"),
                    ("B2", "Correlation Matrix\nFeature Relationships"),
                    ("B3", "Feature Importance\nScoring"),
                ],
            },
            {
                "name": "cluster_selection",
                "label": "FEATURE SELECTION",
                "fill": "#f3e5f5",
                "nodes": [
                    ("C1", "Correlation Threshold > 0.3"),
                    ("C2", "Clinical Relevance"),
                    ("C3", "Variance Analysis"),
                ],
            },
            {
                "name": "cluster_engineering",
                "label": "FEATURE ENGINEERING",
                "fill": "#c8e6c9",
                "nodes": [
                    ("D1", "Composite Features"),
                    ("D2", "Symptom Clustering\nSeverity Groups"),
                    ("D3", "Age Categorization\nChild / Adult / Elderly"),
                    ("D4", "Risk Score Calculation"),
                ],
            },
            {
                "name": "cluster_final",
                "label": "FINAL FEATURE SET",
                "fill": "#ffccbc",
                "nodes": [
                    ("E1", "Selected Features"),
                    ("E2", "Engineered Features"),
                    ("E3", "Optimized Dataset\nReady for Training"),
                ],
            },
        ],
        "extra_nodes": [
            {"id": "F", "label": "Excluded", "shape": "diamond", "style": "filled", "fillcolor": "#fbe9e7"},
        ],
        "edges": [
            ("A1", "B1"),
            ("A2", "B1"),
            ("B1", "B2"),
            ("B2", "B3"),
            ("B3", "C1"),
            ("C1", "C2"),
            ("C2", "C3"),
            ("C3", "D1"),
            ("D1", "D2"),
            ("D2", "D3"),
            ("D3", "D4"),
            ("D4", "E1"),
            ("E1", "E2"),
            ("E2", "E3"),
            ("C1", "F", "fail"),
            ("C2", "F", "fail"),
            ("C3", "F", "fail"),
        ],
    },
    "data_cleaning_pipeline": {
        "rankdir": "TB",
        "clusters": [],
        "extra_nodes": [
            {"id": "Start", "label": "Start: Raw Dataset\n2000 Records", "shape": "oval", "fillcolor": "#4CAF50", "fontcolor": "#ffffff"},
            {"id": "Load", "label": "Load Data\nCSV / Excel", "fillcolor": "#2196F3", "fontcolor": "#ffffff"},
            {"id": "Inspect", "label": "Initial Inspection\nShape 2000 x 18", "fillcolor": "#64b5f6", "fontcolor": "#ffffff"},
            {"id": "Missing", "label": "Missing Values?", "shape": "diamond", "fillcolor": "#FF9800", "fontcolor": "#ffffff"},
            {"id": "Impute", "label": "Median / Mode\nImputation", "fillcolor": "#ffcc80"},
            {"id": "Duplicates", "label": "Duplicate Records?", "shape": "diamond", "fillcolor": "#FF9800", "fontcolor": "#ffffff"},
            {"id": "Remove", "label": "Remove Duplicates\nKeep First", "fillcolor": "#ffe082"},
            {"id": "Encode", "label": "Encode Categorical\nGender Label, Symptoms Binary", "fillcolor": "#ffe0b2"},
            {"id": "Outliers", "label": "Check Outliers", "shape": "diamond", "fillcolor": "#FF9800", "fontcolor": "#ffffff"},
            {"id": "Handle", "label": "Handle Outliers\nIQR / Winsorization", "fillcolor": "#ffd180"},
            {"id": "Scale", "label": "Feature Scaling\nStandardScaler", "fillcolor": "#b3e5fc"},
            {"id": "Split", "label": "Data Splitting\nTrain 80% / Test 20%", "fillcolor": "#c5cae9"},
            {"id": "Validate", "label": "Cross-Validation\n5-Fold", "fillcolor": "#d7ccc8"},
            {"id": "Balance", "label": "Class Imbalance?", "shape": "diamond", "fillcolor": "#FF9800", "fontcolor": "#ffffff"},
            {"id": "SMOTE", "label": "Apply SMOTE", "fillcolor": "#ffe082"},
            {"id": "Ready", "label": "Clean Dataset Ready\nfor Training", "shape": "oval", "fillcolor": "#4CAF50", "fontcolor": "#ffffff"},
        ],
        "edges": [
            ("Start", "Load"),
            ("Load", "Inspect"),
            ("Inspect", "Missing"),
            ("Missing", "Impute"),
            ("Impute", "Duplicates"),
            ("Missing", "Duplicates"),
            ("Duplicates", "Remove"),
            ("Remove", "Encode"),
            ("Duplicates", "Encode"),
            ("Encode", "Outliers"),
            ("Outliers", "Handle"),
            ("Handle", "Scale"),
            ("Outliers", "Scale"),
            ("Scale", "Split"),
            ("Split", "Validate"),
            ("Validate", "Balance"),
            ("Balance", "SMOTE"),
            ("SMOTE", "Ready"),
            ("Balance", "Ready"),
        ],
    },
    "model_training_flow": {
        "rankdir": "LR",
        "clusters": [
            {
                "name": "cluster_data",
                "label": "PREPARED DATA",
                "fill": "#e8f5e9",
                "nodes": [
                    ("A", "Training Set\n1600 Records / 80%"),
                    ("B", "Test Set\n400 Records / 20%"),
                ],
            },
            {
                "name": "cluster_train",
                "label": "MODEL TRAINING",
                "fill": "#fff3e0",
                "nodes": [
                    ("C1", "Decision Tree Training"),
                    ("C2", "SVM Training"),
                    ("C3", "Logistic Regression Training"),
                ],
            },
            {
                "name": "cluster_opt",
                "label": "HYPERPARAMETER TUNING",
                "fill": "#f3e5f5",
                "nodes": [
                    ("D1", "GridSearchCV\n5-Fold"),
                    ("D2", "Parameter Optimization"),
                    ("D3", "Best Parameters Selection"),
                ],
            },
            {
                "name": "cluster_eval",
                "label": "MODEL EVALUATION",
                "fill": "#e1f5fe",
                "nodes": [
                    ("E1", "Accuracy Calculation"),
                    ("E2", "Precision / Recall / F1"),
                    ("E3", "Confusion Matrix"),
                    ("E4", "ROC-AUC Curve"),
                ],
            },
            {
                "name": "cluster_results",
                "label": "RESULTS COMPARISON",
                "fill": "#fce4ec",
                "nodes": [
                    ("F1", "Model Performance Comparison"),
                    ("F2", "Best Model Selection\nSVM 88.2%"),
                    ("F3", "Final Model Deployment"),
                ],
            },
        ],
        "edges": [
            ("A", "C1"),
            ("A", "C2"),
            ("A", "C3"),
            ("C1", "D1"),
            ("C2", "D1"),
            ("C3", "D1"),
            ("D1", "D2"),
            ("D2", "D3"),
            ("D3", "E1"),
            ("B", "E1"),
            ("E1", "E2"),
            ("E2", "E3"),
            ("E3", "E4"),
            ("E4", "F1"),
            ("F1", "F2"),
            ("F2", "F3"),
        ],
    },
}


def apply_theme(dot: Digraph, spec: DiagramSpec) -> None:
    dot.attr(
        rankdir=spec.get("rankdir", "LR"),
        splines=spec.get("splines", "spline"),
        fontname=THEME["font_bold"],
        labelloc="t",
        fontsize="18",
        pad="0.7",
        ranksep="0.9",
        nodesep="0.6",
        bgcolor=THEME["background"],
    )
    dot.attr("node", **THEME["node"])
    dot.attr("edge", **THEME["edge"])


def add_clusters(dot: Digraph, spec: DiagramSpec) -> None:
    for idx, cluster in enumerate(spec.get("clusters", [])):
        fill = cluster.get("fill")
        border_color = cluster.get("border_color", THEME["cluster_border"])
        label_color = cluster.get("label_color", THEME["cluster_label_color"])
        with dot.subgraph(name=cluster["name"]) as c:
            c.attr(
                label=cluster["label"],
                color=border_color,
                style="rounded,filled,setlinewidth(2)",
                fillcolor=fill or f"/pastel28/{(idx % 8) + 1}",
                fontname=THEME["font_bold"],
                fontsize="13",
                fontcolor=label_color,
                labeljust="l",
                margin="18",
            )
            node_style = {
                "shape": cluster.get("shape", THEME["node"]["shape"]),
                "style": cluster.get("style", THEME["node"]["style"]),
                "fillcolor": cluster.get("node_fill", "#ffffff"),
            }
            for node_id, label in cluster["nodes"]:
                c.node(node_id, label, **node_style)


def add_extra_nodes(dot: Digraph, spec: DiagramSpec) -> None:
    for extra in spec.get("extra_nodes", []):
        attrs = {
            "shape": extra.get("shape", "rect"),
            "style": extra.get("style", "rounded,filled,setlinewidth(2)"),
            "fillcolor": extra.get("fillcolor", "#ffffff"),
            "fontcolor": extra.get("fontcolor", "#0f172a"),
            "color": extra.get("border_color", "#94a3b8"),
        }
        dot.node(extra["id"], extra["label"], **attrs)


def add_edges(dot: Digraph, spec: DiagramSpec) -> None:
    for edge in spec.get("edges", []):
        attrs = {}
        if len(edge) == 3:
            if edge[2] == "fail":
                attrs.update({"style": "dashed", "color": "#ef4444"})
            elif isinstance(edge[2], dict):
                attrs.update(edge[2])
        dot.edge(edge[0], edge[1], **attrs)


def render_diagram(name: str, spec: DiagramSpec, output_dir: Path) -> None:
    dot = Digraph(name=name)
    apply_theme(dot, spec)
    add_clusters(dot, spec)
    add_extra_nodes(dot, spec)
    add_edges(dot, spec)

    output_path = output_dir / name
    for fmt in OUTPUT_FORMATS:
        dot.format = fmt
        dot.render(str(output_path), cleanup=True)
        print(f"Diagram saved to {output_path.with_suffix(f'.{fmt}').resolve()}")


def main() -> None:
    ensure_graphviz_binary()
    output_dir = Path("images") / "new"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, spec in DIAGRAMS.items():
        render_diagram(name, spec, output_dir)


if __name__ == "__main__":
    main()

