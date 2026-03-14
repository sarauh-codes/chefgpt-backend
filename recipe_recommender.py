import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np


class RecipeRecommender:
    def __init__(self, csv_path='RECIPES.csv'):
        """Initialize the recommender with dataset and model"""
        # Load dataset
        self.df = pd.read_csv(csv_path, encoding='utf-8-sig')
        self.df.columns = self.df.columns.str.strip()  # Clean column names

        # Add recipe_id column
        self.df.insert(0, 'recipe_id', range(1, len(self.df) + 1))

        # Load pre-trained model (all-MiniLM-L6-v2)
        print("Loading ML model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode all recipe ingredients
        print("Encoding recipe ingredients...")
        self.recipe_embeddings = self.model.encode(
            self.df['ingredients'].tolist(),
            show_progress_bar=True
        )

        # Build BM25 index for keyword matching
        print("Building BM25 keyword index...")
        tokenized_ingredients = [
            ingredients.lower().replace(',', ' ').split() 
            for ingredients in self.df['ingredients'].tolist()
        ]
        self.bm25 = BM25Okapi(tokenized_ingredients)

        print("Model ready! Loaded {} recipes.".format(len(self.df)))

    def recommend(self, user_ingredients, top_k=5, alpha=0.5, min_overlap=0):
        """
        Recommend recipes using hybrid search (semantic + keyword matching)

        Args:
            user_ingredients (str): Comma-separated ingredients (e.g., "chicken, rice, egg")
            top_k (int): Number of recommendations to return
            alpha (float): Weight for BM25 keyword matching (0=pure semantic, 1=pure keyword)
                          Recommended: 0.3-0.5 for balanced results
            min_overlap (int): Minimum number of overlapping ingredients required (0=no filter)

        Returns:
            list: Top recommended recipes with details
        """
        # Encode user ingredients for semantic search
        user_embedding = self.model.encode([user_ingredients])

        # Calculate semantic similarity
        semantic_scores = cosine_similarity(user_embedding, self.recipe_embeddings)[0]

        # Calculate BM25 keyword scores
        tokenized_query = user_ingredients.lower().replace(',', ' ').split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        # Normalize both scores to 0-1 range
        semantic_normalized = (semantic_scores - semantic_scores.min()) / (
            semantic_scores.max() - semantic_scores.min() + 1e-10
        )
        bm25_normalized = (bm25_scores - bm25_scores.min()) / (
            bm25_scores.max() - bm25_scores.min() + 1e-10
        )

        # Combine scores with weighted average
        hybrid_scores = alpha * bm25_normalized + (1 - alpha) * semantic_normalized

        # Apply ingredient overlap filter if specified
        if min_overlap > 0:
            user_ingredients_set = set(tokenized_query)
            overlap_counts = []

            for ingredients in self.df['ingredients'].tolist():
                recipe_ingredients = set(ingredients.lower().replace(',', ' ').split())
                overlap = len(user_ingredients_set & recipe_ingredients)
                overlap_counts.append(overlap)

            # Set scores to -1 for recipes below minimum overlap
            overlap_counts = np.array(overlap_counts)
            hybrid_scores = np.where(overlap_counts >= min_overlap, hybrid_scores, -1)

        # Get top K indices
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]

        # Filter out any recipes with -1 scores (didn't meet min_overlap)
        top_indices = [idx for idx in top_indices if hybrid_scores[idx] >= 0][:top_k]

        # Prepare recommendations
        recommendations = []
        for idx in top_indices:
            recipe = self.df.iloc[idx]

            # Calculate ingredient overlap for display
            user_ingredients_set = set(tokenized_query)
            recipe_ingredients = set(recipe['ingredients'].lower().replace(',', ' ').split())
            overlap = len(user_ingredients_set & recipe_ingredients)

            recommendations.append({
                'recipe_id': int(recipe['recipe_id']),
                'recipe_name': recipe['recipe_name'],
                'ingredients': recipe['ingredients'],
                'cuisine': recipe['cuisine'],
                'calories': int(recipe['calories']),
                'rating': float(recipe['rating']),
                'difficulty': recipe.get('difficulty', 'medium'),
                'similarity_score': round(float(hybrid_scores[idx]) * 100, 2),  # Hybrid score
                'semantic_score': round(float(semantic_normalized[idx]) * 100, 2),  # For debugging
                'keyword_score': round(float(bm25_normalized[idx]) * 100, 2),  # For debugging
                'ingredient_overlap': overlap
            })

        return recommendations

    def recommend_semantic_only(self, user_ingredients, top_k=5):
        """
        Original semantic-only recommendation (for comparison)
        """
        return self.recommend(user_ingredients, top_k=top_k, alpha=0.0, min_overlap=0)

    def recommend_keyword_only(self, user_ingredients, top_k=5):
        """
        Keyword-only recommendation using BM25
        """
        return self.recommend(user_ingredients, top_k=top_k, alpha=1.0, min_overlap=0)

    def get_recipe_by_id(self, recipe_id):
        """Get full recipe details by ID"""
        recipe = self.df[self.df['recipe_id'] == recipe_id]
        if len(recipe) == 0:
            return None

        recipe = recipe.iloc[0]

        # Check if instructions column exists, if not use placeholder
        instructions = recipe.get('instructions', 'Instructions will be added soon.')

        return {
            'recipe_id': int(recipe['recipe_id']),
            'recipe_name': recipe['recipe_name'],
            'ingredients': recipe['ingredients'],
            'cuisine': recipe['cuisine'],
            'calories': int(recipe['calories']),
            'rating': float(recipe['rating']),
            'difficulty': recipe.get('difficulty', 'medium'),
            'instructions': instructions
        }


# Initialize recommender globally (so it loads once when app starts)
recommender = None

def get_recommender():
    """Get or initialize the recommender"""
    global recommender
    if recommender is None:
        recommender = RecipeRecommender()
    return recommender
