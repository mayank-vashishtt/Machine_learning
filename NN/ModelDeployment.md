# Session 5: Model Deployment — Detailed Notes

## 1. What is Model Deployment?

* **Definition:** The process of making a trained ML model available for real-world use by users or applications, so it can generate predictions in real time (production).
* **Importance:** Deployment transforms a research/experimental model into a practical, usable service.
* **Examples:** Powering recommendation engines (Netflix), fraud detection systems (banks), chatbots, and healthcare diagnosis tools.

---

## 2. Local Model Serving — Flask

### What is Flask?

* A lightweight Python web framework.
* Allows creation of APIs to serve models: take input → process with model → return output.

### Basic Workflow:

1. **Train & Save Model:** Train in Python; save to disk using Pickle, Joblib, or similar.
2. **Create Flask App:**

   * Load model on startup.
   * Define API endpoints (`/predict`) to accept requests and send back predictions.
3. **Run Flask Server:**

   * Receives JSON input from clients.
   * Calls the model.
   * Returns prediction as JSON.

**Analogy:** Flask acts like a receptionist: collects inputs, queries the model, delivers results.

---

## 3. Model Deployment with Containers — Docker

### What is Docker?

* A tool that **packages your app + model + dependencies + environment** into a **container**.
* Guarantees consistency across different machines.

### Why Use Docker?

* **Portability:** Works across Windows, Mac, Linux.
* **Reproducibility:** Same setup for devs, testers, servers.
* **Isolation:** Keeps dependencies clean, avoids conflicts.

### Model-in-Container Flow:

1. Bundle Flask app + model into a **Docker image**.
2. Run image as a **container**.
3. Flask API is now portable and can run anywhere Docker runs.

---

## 4. Cloud-Scale Deployment — AWS ECS (Elastic Container Service)

### What is AWS ECS?

* Amazon’s managed service to run Docker containers at scale.
* Handles orchestration: launching, updating, monitoring, scaling.

### Workflow for ML Models:

1. Push Docker image to **AWS ECR** (Elastic Container Registry).
2. Configure ECS to run the container (CPU, memory, scaling rules).
3. ECS deploys containers on AWS infrastructure.
4. Endpoint exposed → users/apps can call Flask API over the internet.

### Benefits:

* **Scalability:** Auto-scales with demand.
* **Reliability:** Handles failures automatically.
* **Maintainability:** Easy to push new versions of your model.

---

## 5. General Steps (End-to-End Deployment)

1. Train and save model locally.
2. Build Flask API for predictions.
3. Dockerize the Flask app.
4. Test Docker container locally.
5. Push container image to AWS ECR.
6. Deploy container via AWS ECS.
7. Expose public endpoint for users.

---

## 6. Summary Table

| Step            | Tool          | Purpose                          |
| --------------- | ------------- | -------------------------------- |
| Save Model      | Pickle/Joblib | Store trained ML model           |
| Serve API       | Flask         | Provide predictions via HTTP     |
| Containerize    | Docker        | Bundle model + app + environment |
| Cloud Registry  | AWS ECR       | Store Docker images              |
| Scalable Deploy | AWS ECS       | Run & manage models in cloud     |

---

⚡ **Key Insight:** Flask makes your model accessible, Docker makes it portable, and AWS ECS makes it scalable.
