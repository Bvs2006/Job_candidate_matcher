import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

candidates = pd.read_csv("candidates_200.csv")
jobs = pd.read_csv("jobs_200.csv")


stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z, ]', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

candidates['profile'] = candidates['skills'] + " " + candidates['education']
candidates['profile'] = candidates['profile'].apply(clean_text)


vectorizer = TfidfVectorizer()
all_profiles = list(candidates['profile']) + list(jobs['required_skills'])
tfidf_matrix = vectorizer.fit_transform(all_profiles)


k = 2  # 1 cluster for Data Science, 1 for Web Dev
kmeans = KMeans(n_clusters=k, random_state=42)
candidate_vectors = tfidf_matrix[:len(candidates)]
candidate_labels = kmeans.fit_predict(candidate_vectors)
candidates['cluster'] = candidate_labels


job_vectors = tfidf_matrix[len(candidates):]
similarity = cosine_similarity(candidate_vectors, job_vectors)

# Example: top job for each candidate
for i, row in enumerate(similarity):
    top_job_idx = row.argmax()
    print(candidates.loc[i,'name'], "→", jobs.loc[top_job_idx,'job_title'])

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
reduced = pca.fit_transform(candidate_vectors.toarray())
plt.scatter(reduced[:,0], reduced[:,1], c=candidate_labels)
plt.show()
