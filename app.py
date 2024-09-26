import os
import requests
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__, template_folder='/Users/shreyanshjain/PycharmProjects/food_recommendation_system/templates')

# Load dataset
data = pd.read_csv("recipe_final (1).csv")


# Preprocess Ingredients (same as before)
def clean_ingredients(ingredients):
    ingredients = ingredients.lower()
    ingredients = re.sub(r'[^\w\s]', '', ingredients)
    ingredients = re.sub(r'\b(oz|cup|cups|tablespoon|tablespoons|teaspoon|teaspoons|gram|grams|ml|litre|litres)\b', '',
                         ingredients)
    return ingredients


data['cleaned_ingredients'] = data['ingredients_list'].apply(clean_ingredients)

# Vectorizer and KNN model setup (same as before)
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=0.01)
X_ingredients = vectorizer.fit_transform(data['cleaned_ingredients'])

scaler = StandardScaler()
X_numerical = scaler.fit_transform(
    data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])

X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)


# Function to get recommendations from Spoonacular API
def get_spoonacular_recipes(ingredients):
    api_key = os.getenv('dba3dc0a4978422abcd5ec47bada1c2a')  # Use environment variable for API key
    url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={ingredients}&number=5&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []


# Combine both dataset and API recommendations
def recommend_recipes(input_features):
    # Check if nutritional values are provided
    if len(input_features) > 7 and all(input_features[:7]):
        # Local dataset recommendations with numerical and ingredient features
        input_features_scaled = scaler.transform([input_features[:7]])
    else:
        # Only ingredient-based recommendations (ignore numerical features)
        input_features_scaled = np.zeros((1, 7))  # Fill with zeros for missing nutritional values

    cleaned_input_ingredients = clean_ingredients(input_features[7])
    input_ingredients_transformed = vectorizer.transform([cleaned_input_ingredients])
    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])
    distances, indices = knn.kneighbors(input_combined)

    dataset_recommendations = data.iloc[indices[0]][['recipe_name', 'ingredients_list', 'image_url']].head(5).to_dict(
        orient='records')

    # Spoonacular API recommendations
    api_recommendations = get_spoonacular_recipes(input_features[7])

    # Format API recommendations to match local data format
    api_recommendation_formatted = [
        {
            'recipe_name': recipe['title'],
            'ingredients_list': ', '.join([i['name'] for i in recipe['usedIngredients'] + recipe['missedIngredients']]),
            'image_url': recipe['image']
        }
        for recipe in api_recommendations
    ]

    # Combine both results
    return dataset_recommendations + api_recommendation_formatted


# Route and form handling
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # If the user doesn't input values, handle missing values
            calories = float(request.form['calories']) if request.form['calories'] else 0
            fat = float(request.form['fat']) if request.form['fat'] else 0
            carbohydrates = float(request.form['carbohydrates']) if request.form['carbohydrates'] else 0
            protein = float(request.form['protein']) if request.form['protein'] else 0
            cholesterol = float(request.form['cholesterol']) if request.form['cholesterol'] else 0
            sodium = float(request.form['sodium']) if request.form['sodium'] else 0
            fiber = float(request.form['fiber']) if request.form['fiber'] else 0
            ingredients = request.form['ingredients']

            input_features = [calories, fat, carbohydrates, protein, cholesterol, sodium, fiber, ingredients]
            recommendations = recommend_recipes(input_features)
            return render_template('index.html', recommendations=recommendations)

        except ValueError:
            # Handle any errors in form input
            return render_template('index.html', recommendations=[], error="Please provide valid inputs.")

    return render_template('index.html', recommendations=[])


if __name__ == '__main__':
    app.run(debug=True)
