import os
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, template_folder='/Users/shreyanshjain/PycharmProjects/food_recommendation_system/templates')
app.secret_key = os.getenv('SECRET_KEY', 'your_secret_key')  # Ensure secret key is loaded
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['TEMPLATES_AUTO_RELOAD'] = True
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)


# Load dataset and preprocess
data = pd.read_csv("recipe_final (1).csv")


def clean_ingredients(ingredients):
    ingredients = ingredients.lower()
    ingredients = re.sub(r'[^\w\s]', '', ingredients)
    return ingredients


data['cleaned_ingredients'] = data['ingredients_list'].apply(clean_ingredients)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, min_df=0.01)
X_ingredients = vectorizer.fit_transform(data['cleaned_ingredients'])
scaler = StandardScaler()
X_numerical = scaler.fit_transform(
    data[['calories', 'fat', 'carbohydrates', 'protein', 'cholesterol', 'sodium', 'fiber']])
X_combined = np.hstack([X_numerical, X_ingredients.toarray()])

knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
knn.fit(X_combined)


def get_spoonacular_recipes(ingredients):
    api_key = os.getenv('SPOONACULAR_API_KEY')
    url = f"https://api.spoonacular.com/recipes/findByIngredients?ingredients={ingredients}&number=5&apiKey={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []


def recommend_recipes(input_features):
    # Scaled numerical input
    if len(input_features) > 7 and all(input_features[:7]):
        input_features_scaled = scaler.transform([input_features[:7]])
    else:
        input_features_scaled = np.zeros((1, 7))

    # Clean and vectorize the input ingredients
    cleaned_input_ingredients = clean_ingredients(input_features[7])
    input_ingredients_transformed = vectorizer.transform([cleaned_input_ingredients])
    input_combined = np.hstack([input_features_scaled, input_ingredients_transformed.toarray()])

    # Dataset recommendations
    distances, indices = knn.kneighbors(input_combined)
    dataset_recommendations = data.iloc[indices[0]][['recipe_name', 'ingredients_list', 'image_url']].head(4).to_dict(
        orient='records')

    # Fetch API recommendations
    api_recommendations = get_spoonacular_recipes(input_features[7])

    # Format API recommendations
    api_recommendation_formatted = []
    for recipe in api_recommendations:
        used_ingredients = ', '.join([ingredient['name'] for ingredient in recipe.get('usedIngredients', [])])
        missed_ingredients = ', '.join([ingredient['name'] for ingredient in recipe.get('missedIngredients', [])])
        all_ingredients = f"Used: {used_ingredients}. Missed: {missed_ingredients}" if used_ingredients and missed_ingredients else used_ingredients or missed_ingredients

        api_recommendation_formatted.append({
            'recipe_name': recipe['title'],
            'ingredients_list': all_ingredients,
            'image_url': recipe['image']
        })

    # Combine dataset and API recommendations (limit to 7 total recommendations)
    combined_recommendations = (dataset_recommendations + api_recommendation_formatted)[:7]

    return combined_recommendations


@app.route('/', methods=['GET'])
def root():
    return redirect(url_for('login'))  # Redirects to the login page


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Change the hashing method to 'pbkdf2:sha256'
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password)

        try:
            db.session.add(new_user)
            db.session.commit()
            flash('Registration successful! You can now log in.', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()  # Rollback if there's an error
            flash('Username already exists. Please choose a different one.', 'danger')

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Extract nutritional values and ingredients from the form
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
            return render_template('index.html', recommendations=[], error="Please provide valid inputs.")

    return render_template('index.html', recommendations=[])


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
