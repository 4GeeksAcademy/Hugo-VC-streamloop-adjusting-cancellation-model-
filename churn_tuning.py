from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_URL = (
    "https://raw.githubusercontent.com/IBM/"
    "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
RANDOM_STATE = 42
TEST_SIZE = 0.2
F2_SCORER = make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0)


def build_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessing),
            ("classifier", LogisticRegression()),
        ]
    )


def evaluate_once(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict:
    churn_probabilities = model.predict_proba(features)[:, 1]
    predictions = (churn_probabilities >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "f2": fbeta_score(target, predictions, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(target, churn_probabilities),
        "confusion_matrix": confusion_matrix(target, predictions, labels=[0, 1]),
    }


def format_metrics(metrics: dict) -> str:
    return "\n".join(
        [
            f"| {name} | {metrics[name]:.4f} |"
            for name in ("accuracy", "precision", "recall", "f1", "f2", "roc_auc")
        ]
    )


def format_candidates(results: pd.DataFrame, count: int = 5) -> str:
    rows = [
        "| Puesto | F2 medio | Desv. estándar | Hiperparámetros |",
        "| ---: | ---: | ---: | --- |",
    ]
    top = results.sort_values(
        ["mean_test_score", "std_test_score"], ascending=[False, True]
    ).head(count)
    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        params = json.dumps(row["params"], ensure_ascii=True, sort_keys=True)
        rows.append(
            f"| {rank} | {row['mean_test_score']:.4f} | "
            f"{row['std_test_score']:.4f} | `{params}` |"
        )
    return "\n".join(rows)


def main() -> None:
    data = pd.read_csv(DATA_URL)
    initial_rows = len(data)
    data = data.drop_duplicates().copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    target = data.pop("Churn").map({"No": 0, "Yes": 1})
    if target.isna().any():
        raise ValueError("La columna Churn contiene etiquetas desconocidas o vacías.")
    features = data.drop(columns=["customerID"], errors="ignore")

    numeric_columns = features.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=np.number).columns.tolist()
    missing_by_column = features.isna().sum()
    missing_by_column = missing_by_column[missing_by_column > 0].to_dict()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_pipeline(numeric_columns, categorical_columns)

    baseline = pipeline.fit(X_train, y_train)
    baseline_metrics = evaluate_once(baseline, X_test, y_test)

    randomized_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions={
            "classifier__C": loguniform(0.01, 100),
            "classifier__class_weight": [None, "balanced"],
        },
        n_iter=16,
        scoring=F2_SCORER,
        cv=cv,
        refit=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        return_train_score=False,
    )
    randomized_search.fit(X_train, y_train)
    random_results = pd.DataFrame(randomized_search.cv_results_)

    top_random = random_results.sort_values(
        ["mean_test_score", "std_test_score"], ascending=[False, True]
    ).head(3)
    randomized_c_values = top_random["param_classifier__C"].astype(float).tolist()
    c_values = sorted(
        {
            round(value * factor, 8)
            for value in randomized_c_values
            for factor in (0.5, 1.0, 2.0)
        }
    )
    class_weights = sorted(
        {str(value) for value in top_random["param_classifier__class_weight"]}
    )
    weight_values = [None if value == "None" else value for value in class_weights]

    def select_stable_candidate(cv_results: dict) -> int:
        stability_adjusted_scores = (
            cv_results["mean_test_score"] - 0.5 * cv_results["std_test_score"]
        )
        return int(np.nanargmax(stability_adjusted_scores))

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "classifier__C": c_values,
            "classifier__class_weight": weight_values,
        },
        scoring=F2_SCORER,
        cv=cv,
        refit=select_stable_candidate,
        n_jobs=-1,
        return_train_score=False,
    )
    grid_search.fit(X_train, y_train)
    grid_results = pd.DataFrame(grid_search.cv_results_)
    final_metrics = evaluate_once(grid_search.best_estimator_, X_test, y_test)
    selected_cv = grid_results.iloc[grid_search.best_index_]
    randomized_best_cv = random_results.iloc[randomized_search.best_index_]

    test_churn_rate = y_test.mean()
    report = f"""# StreamLoop: informe de ajuste del modelo de churn

## Datos y preparación

- Fuente: [dataset Telco Customer Churn de IBM]({DATA_URL}).
- Filas originales: {initial_rows}; filas tras eliminar duplicados exactos: {len(data)}.
- Predictores: {features.shape[1]} (se excluyó `customerID`, identificador sin valor predictivo); filas de entrenamiento: {len(X_train)}; filas de test: {len(X_test)}.
- Proporción de churn en test: {test_churn_rate:.1%}.
- Valores faltantes antes del ajuste: `{json.dumps(missing_by_column, ensure_ascii=True, sort_keys=True)}`. `TotalCharges` se convierte a numérico; los faltantes se imputan dentro del pipeline (mediana en numéricas, moda en categóricas). Escalado numérico y codificación one-hot también se ajustan dentro del pipeline.
- División estratificada: 80/20, `random_state={RANDOM_STATE}`. Validación cruzada estratificada de 5 folds sobre entrenamiento.

## Métrica y criterio de selección

La métrica principal es F2 para la clase positiva `Churn=Yes`: pondera más el recall que la precisión, alineándose con el coste de no detectar una cancelación. Se reportan también precisión, recall, F1, F2, ROC-AUC y matriz de confusión. La matriz sigue el orden `[No, Yes]` en filas reales y columnas predichas; el umbral de clasificación es 0.5.

El candidato final de la búsqueda en grilla se selecciona maximizando `F2 medio - 0.5 × desviación estándar entre folds`. Así se favorece el recall sin ignorar la estabilidad, aunque el criterio puede preferir un promedio algo menor.

## Búsquedas

Se utilizó regresión logística como clasificador de referencia. La línea base conserva sus hiperparámetros por defecto. Ambas búsquedas ajustan únicamente `X_train` e incluyen el preprocesamiento en el pipeline; el test no participa en el ajuste.

- `RandomizedSearchCV`: `C ~ loguniform(0.01, 100)`, `class_weight ∈ {{None, 'balanced'}}`, 16 candidatos, 5 folds y `n_jobs=-1`.
- `GridSearchCV`: refinamiento alrededor de los valores de `C` de los tres mejores candidatos aleatorios; `class_weight` se restringe a las opciones observadas entre esos candidatos; 5 folds y `n_jobs=-1`.

Mejor candidato aleatorio: `{json.dumps(randomized_search.best_params_, ensure_ascii=True, sort_keys=True)}`; F2 medio `{randomized_best_cv['mean_test_score']:.4f}` ± `{randomized_best_cv['std_test_score']:.4f}`.

### Mejores candidatos aleatorios por F2 medio

{format_candidates(random_results)}

### Mejores candidatos de grilla por F2 medio

{format_candidates(grid_results)}

## Evaluación en test

El test se evaluó exactamente dos veces: una para la línea base y otra para el pipeline final ya reajustado por `GridSearchCV` (`refit` con criterio estable). No se volvió a ajustar manualmente el estimador final.

| Métrica | Línea base | Modelo final |
| --- | ---: | ---: |
"""
    for metric in ("accuracy", "precision", "recall", "f1", "f2", "roc_auc"):
        report += (
            f"| {metric} | {baseline_metrics[metric]:.4f} | "
            f"{final_metrics[metric]:.4f} |\n"
        )
    report += f"""
| Matriz de confusión `[[TN, FP], [FN, TP]]` | `{baseline_metrics['confusion_matrix'].tolist()}` | `{final_metrics['confusion_matrix'].tolist()}` |

## Modelo elegido y lectura del resultado

- Hiperparámetros: `{json.dumps(grid_search.best_params_, ensure_ascii=True, sort_keys=True)}`.
- F2 de validación cruzada del candidato seleccionado: `{selected_cv['mean_test_score']:.4f} ± {selected_cv['std_test_score']:.4f}`; criterio ajustado por estabilidad: `{selected_cv['mean_test_score'] - 0.5 * selected_cv['std_test_score']:.4f}`.
- Cambio de recall en test frente a la línea base: `{final_metrics['recall'] - baseline_metrics['recall']:+.4f}`; cambio de precisión: `{final_metrics['precision'] - baseline_metrics['precision']:+.4f}`.

Un aumento del recall suele recuperar más clientes en riesgo, pero puede generar más falsos positivos y contactos de retención innecesarios. La matriz de confusión y el cambio en precisión permiten valorar ese coste junto con el objetivo principal. Las métricas de test sirven para comparar estos dos modelos y no se usaron para elegir hiperparámetros.
"""
    Path("tuning_report.md").write_text(report, encoding="utf-8")

    print(f"Reporte guardado en tuning_report.md")
    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    print(
        "Test (baseline/final) F2: "
        f"{baseline_metrics['f2']:.4f} / {final_metrics['f2']:.4f}; "
        "recall: "
        f"{baseline_metrics['recall']:.4f} / {final_metrics['recall']:.4f}"
    )


if __name__ == "__main__":
    main()