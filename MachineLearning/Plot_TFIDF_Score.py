import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Define a sample document
document = [
    'Python is a high-level, interpreted programming language known for its readability and versatility.',
    'Python libraries are collections of pre-written code that users can import and use in their programs.',
    'List comprehension is a concise way to create lists in Python, allowing you to generate new lists based on existing lists or other iterable objects using a compact syntax.',
    'Object-oriented programming (OOP) is a programming paradigm that uses objects and classes to organize and structure code, focusing on creating reusable and modular software design.',
    'Functional programming is a programming paradigm that treats computation as the evaluation of mathematical functions, emphasizing immutable data, pure functions, and avoiding changing state and mutable data.'    
]

# Compute TF-IDF scores
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(document)

# Get feature names (words) and their TF-IDF scores
feature_names = vectorizer.get_feature_names_out()
tfidf_scores = tfidf_matrix.toarray()[0]  # Use the first document for visualization

# Select a few words to visualize
words_to_visualize = ["python", "functions","programs", "code", "data"]
tfidf_values = [tfidf_scores[feature_names.tolist().index(word)] for word in words_to_visualize]

# Create the bar chart
plt.figure(figsize=(10, 6))
plt.bar(words_to_visualize, tfidf_values, color='skyblue')
plt.xlabel('Words', fontsize=12)
plt.ylabel('TF-IDF Score', fontsize=12)
plt.title('TF-IDF Scores for Selected Words', fontsize=14)
plt.ylim(0, 1)  # Set y-axis limit to 1 for better visualization
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.savefig('tfidf_scores.png', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
