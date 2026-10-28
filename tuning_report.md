# StreamLoop: informe de ajuste del modelo de churn

## Datos y preparación

- Fuente: [dataset Telco Customer Churn de IBM](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
- Filas originales: 7043; tras quitar duplicados: 7043; predictores: 19 (se excluyó `customerID`).
- Entrenamiento: 5634 filas; test: 1409 filas; proporción churn en test: 26.5%.
- Faltantes antes del pipeline: `{"TotalCharges": 11}`. `TotalCharges` se convierte a numérico; imputación, escalado y codificación se ajustan dentro del pipeline.
- División estratificada 80/20 (`random_state=42`); validación cruzada estratificada de 5 folds.

## Métrica y método

La métrica de búsqueda es F2 para `Churn=Yes` (`beta=2`), que prioriza recall por el coste de no detectar cancelaciones. El candidato final maximiza `F2 medio - 0.5 × desviación estándar entre folds`.

- RandomizedSearchCV: 16 candidatos, `C ~ loguniform(0.01, 100)`, `class_weight` en `(None, 'balanced')`, `n_jobs=-1`.
- GridSearchCV: refinamiento de `C` alrededor de los tres mejores candidatos aleatorios y pesos observados; `n_jobs=-1`.
- Mejor candidato aleatorio: `{"classifier__C": 0.21618942406574443, "classifier__class_weight": "balanced"}`; F2 CV `0.7216 ± 0.0288`.

### Candidatos aleatorios

| Puesto | F2 medio | Desv. estándar | Hiperparámetros |
| ---: | ---: | ---: | --- |
| 1 | 0.7216 | 0.0288 | `{"classifier__C": 0.21618942406574443, "classifier__class_weight": "balanced"}` |
| 2 | 0.7213 | 0.0304 | `{"classifier__C": 6.79657809075816, "classifier__class_weight": "balanced"}` |
| 3 | 0.7204 | 0.0305 | `{"classifier__C": 21.368329072358772, "classifier__class_weight": "balanced"}` |
| 4 | 0.7200 | 0.0253 | `{"classifier__C": 0.03613894271216528, "classifier__class_weight": "balanced"}` |
| 5 | 0.7197 | 0.0258 | `{"classifier__C": 0.05416754583247458, "classifier__class_weight": "balanced"}` |

### Candidatos de grilla

| Puesto | F2 medio | Desv. estándar | Hiperparámetros |
| ---: | ---: | ---: | --- |
| 1 | 0.7228 | 0.0299 | `{"classifier__C": 3.39828905, "classifier__class_weight": "balanced"}` |
| 2 | 0.7219 | 0.0288 | `{"classifier__C": 0.43237885, "classifier__class_weight": "balanced"}` |
| 3 | 0.7216 | 0.0288 | `{"classifier__C": 0.21618942, "classifier__class_weight": "balanced"}` |
| 4 | 0.7213 | 0.0304 | `{"classifier__C": 6.79657809, "classifier__class_weight": "balanced"}` |
| 5 | 0.7207 | 0.0267 | `{"classifier__C": 0.10809471, "classifier__class_weight": "balanced"}` |

## Comparación en test

Test evaluado exactamente dos veces: baseline y modelo final. Búsquedas ajustadas únicamente sobre entrenamiento. Métricas adicionales a F2: precisión, recall, F1, accuracy, ROC-AUC y matriz de confusión.

| Métrica | Línea base | Modelo final |
| --- | ---: | ---: |
| accuracy | 0.8055 | 0.7410 |
| precision | 0.6572 | 0.5078 |
| recall | 0.5588 | 0.7834 |
| f1 | 0.6040 | 0.6162 |
| f2 | 0.5761 | 0.7067 |
| roc_auc | 0.8419 | 0.8410 |

| Matriz `[[TN, FP], [FN, TP]]` | `[[926, 109], [165, 209]]` | `[[751, 284], [81, 293]]` |

## Selección y trade-off

- Hiperparámetros finales: `{"classifier__C": 3.39828905, "classifier__class_weight": "balanced"}`.
- F2 CV seleccionado: `0.7228 ± 0.0299`; criterio de estabilidad: `0.7078`.
- Cambio de recall: `+0.2246`; cambio de precisión: `-0.1494`. El mayor recall reduce churners no detectados, pero la menor precisión puede implicar más ofertas de retención innecesarias.
