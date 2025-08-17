# Session 6: MLOps — Detailed Notes

## 1. What is MLOps?

* **Definition:** MLOps (Machine Learning Operations) combines Machine Learning with DevOps principles to manage the **entire lifecycle** of ML models — from development to deployment to monitoring.
* **Goals:** Ensure automation, reproducibility, scalability, and reliability in ML systems.
* **Why Important?** Moves beyond just training models — focuses on making them **usable, maintainable, and robust in production**.

---

## 2. Data & Model Versioning

### Purpose:

* Keeps track of all changes to **code, data, and models** for reproducibility, auditing, and debugging.
* Enables rollback and ensures full lineage for every prediction.

### Types:

* **Code Versioning:** Git, GitHub, GitLab.
* **Data Versioning:** DVC, LakeFS, MLflow.
* **Model Versioning:** MLflow, DVC, Model Registries.

### Benefits:

* Traceability (know which data + code created which model).
* Debugging becomes easier.
* Enables rollback to older stable versions.
* Auditing and compliance.

---

## 3. CI/CD (Continuous Integration & Deployment)

### Continuous Integration (CI):

* Automates testing and building every time new code or model changes are pushed.
* Ensures quality and prevents broken code/models.

### Continuous Deployment (CD):

* Automates release of tested models to production.
* Enables rapid, safe updates without manual intervention.

### Tools:

* Jenkins, GitHub Actions, GitLab CI/CD, Kubeflow Pipelines, MLflow, Docker.

---

## 4. Model Monitoring & Drift Detection

### Monitoring:

* Tracks model **performance, latency, errors, and prediction quality** in production.
* Ensures reliability and user trust.

### Drift Detection:

* **Data Drift:** Input distribution changes (e.g., user behavior shifts).
* **Concept Drift:** Relationship between inputs and outputs changes.
* Alerts and triggers retraining when drift occurs.

### Tools:

* Prometheus, Grafana, CloudWatch (AWS), MLflow, Evidently AI, Alibi Detect.

---

## 5. Automated Retraining & Continuous Improvement

### Why?

* Models degrade over time as data changes. Continuous improvement ensures relevance.

### Steps:

1. Monitor model performance.
2. Detect drift or schedule retraining.
3. Fetch and preprocess new data.
4. Retrain and evaluate.
5. Deploy if performance improves.
6. Version new model in registry.

### Tools:

* Kubeflow, MLflow, Airflow, AWS SageMaker Pipelines, Jenkins.

---

## 6. MLOps Best Practices

* Track **all artifacts** (code, data, models) for reproducibility.
* Automate testing and deployment (CI/CD pipelines).
* Monitor models continuously in production.
* Automate retraining pipelines.
* Enable rollback to stable versions.
* Foster collaboration between **data scientists, engineers, and ops teams**.

---

## 7. Summary Table: MLOps Lifecycle

| Stage                  | Tools                             | Best Practices                         |
| ---------------------- | --------------------------------- | -------------------------------------- |
| **Versioning**         | Git, DVC, MLflow                  | Track all artifacts                    |
| **CI/CD**              | Jenkins, GitHub Actions, Kubeflow | Automate testing & deployment          |
| **Monitoring & Drift** | Prometheus, Grafana, Evidently    | Live performance checks                |
| **Retraining**         | MLflow, Airflow, SageMaker        | Automated pipelines retrain on trigger |
| **Rollback/Auditing**  | Model Registries                  | Revert instantly, full traceability    |

---

⚡ **Key Takeaway:** MLOps ensures ML systems are **robust, automated, explainable, and continuously improving**. It is not just about training models, but about running them reliably in real-world production environments.
