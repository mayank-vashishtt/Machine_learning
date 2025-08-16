# Session 3: Recommendation Systems — Detailed Notes

## 1. What is a Recommendation System?

A **Recommendation System (RS)** is an algorithm that suggests relevant items (products, movies, music, courses, etc.) to users based on available data.

* **Purpose:** Personalizes user experience, increases user engagement, retention, and sales.
* **Real-world Usage:** Netflix (movies), Amazon (products), Spotify (music), YouTube (videos), LinkedIn (jobs, connections).

---

## 2. Types of Recommendation Systems

### A. Collaborative Filtering

* Uses **user behavior patterns** (ratings, clicks, purchases) without requiring item content.

**1. User–User Collaborative Filtering:**

* Finds users similar to the target user.
* Recommends items liked by those similar users.
* Example: “Users like you also liked...”

**2. Item–Item Collaborative Filtering:**

* Finds items similar to the ones a user already liked.
* Example: “Customers who bought X also bought Y.”

**3. Matrix Factorization (SVD, ALS):**

* Decomposes the **user–item rating matrix** into two smaller matrices:

  * User matrix (users × latent features)
  * Item matrix (items × latent features)
* Multiplying them approximates missing ratings.
* **Latent features** capture hidden traits (e.g., “likes action movies” or “prefers comedy”).

---

### B. Content-Based Filtering

* Recommends items **similar in content** to those a user liked.
* Uses item features (genre, keywords, description, tags).
* Example: If a user likes sci-fi movies, recommend other sci-fi titles.
* **Pros:** Works well with new/small datasets.
* **Cons:** Limited by available metadata, hard to suggest “serendipitous” items.

---

### C. Hybrid Systems

* Combine collaborative + content-based approaches.
* Example: Netflix uses both viewing history (collaborative) and movie attributes (content).
* **Benefits:** Handles cold-start problems, more robust.

---

### D. Classification-Based Systems

* Use classifiers (logistic regression, decision trees, neural networks) to predict if a user will like an item.
* Often integrated into hybrid systems.

---

### E. Other Systems

* **Knowledge-Based:** Recommendations from expert rules/knowledge base.
* **Demographic-Based:** Uses age, gender, location.
* **Sequence-Aware:** Focuses on temporal behavior (next-song prediction in Spotify).
* **Rule-Based:** Manual rules like “if user buys phone → recommend phone cover.”

---

## 3. Matrix Factorization & PCA

### Matrix Factorization:

* Goal: Fill in missing entries of the user–item rating matrix.
* Process:

  * Decompose rating matrix R into User matrix U and Item matrix V.
  * Approximate missing rating: **R ≈ U × Vᵀ**.
* Example: If user likes action and item has “action” weight → high predicted score.

### PCA (Principal Component Analysis):

* A dimensionality reduction technique.
* Keeps only main patterns of user behavior.
* Used to simplify rating data before matrix factorization.
* Reduces noise, speeds computation.

**Key Benefit:** Captures hidden preferences and reduces data sparsity.

---

## 4. General Algorithm Steps

### Collaborative Filtering (Matrix Factorization)

1. Build the ratings matrix (users × items).
2. Preprocess (fill missing values, normalize).
3. Apply SVD/ALS to factorize into latent features.
4. Reconstruct matrix → predict missing ratings.
5. Recommend top-N items for each user.

### Content-Based Filtering

1. Collect item features (e.g., tags, genre, text embeddings).
2. Build user profiles (average of liked items’ features).
3. Compute similarity between user profile and item features (e.g., cosine similarity).
4. Recommend most similar items.

---

## 5. Evaluation Metrics

* **Accuracy-based:** Precision, Recall, F1-score (how correct recommendations are).
* **Ranking-based:** MAP (Mean Average Precision), MRR (Mean Reciprocal Rank), NDCG (Normalized Discounted Cumulative Gain).
* **Beyond Accuracy:**

  * **Coverage:** Fraction of items recommended.
  * **Diversity:** Are recommendations varied?
  * **Novelty:** Are items surprising/new to the user?
  * **Serendipity:** Delightful, unexpected but relevant suggestions.

---

## 6. Model Deployment (Theory)

### A. Using Flask

* Save trained model.
* Build a **Flask API** with endpoints for recommendations.
* Serve predictions via HTTP request.
* Example: `GET /recommend?user_id=123` returns top-N items.

### B. On AWS ECS (Elastic Container Service)

* Package Flask app inside **Docker container**.
* Push container to **Amazon ECR**.
* Deploy using **ECS**, which auto-scales with traffic.
* Users access app via public URL.

---

## 7. Key Points for Exams/Papers

* Define and compare all types of recommendation systems.
* Be able to explain collaborative vs content-based filtering.
* Explain matrix factorization & PCA with examples.
* Understand evaluation metrics (accuracy vs diversity).
* Know deployment basics: Flask, Docker, AWS ECS.

---

⚡ Recommendation Systems are at the heart of modern AI applications, enabling personalization at scale and driving user engagement across industries.