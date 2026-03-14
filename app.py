import os

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_bcrypt import Bcrypt
from datetime import datetime
from recipe_recommender import get_recommender

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///chefgpt.db"
app.config["SECRET_KEY"] = "chefgpt-secret-key-2025"

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# User database model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class CookedRecipe(db.Model):
    __tablename__ = 'cooked_recipes'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipe_id = db.Column(db.Integer, nullable=False)
    recipe_name = db.Column(db.String(200), nullable=False)
    cooked_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to user
    user = db.relationship('User', backref='cooked_recipes')

class Feedback(db.Model):
    __tablename__ = 'feedbacks'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipe_id = db.Column(db.Integer, nullable=False)
    recipe_name = db.Column(db.String(200), nullable=False)
    rating = db.Column(db.Integer, nullable=True)  # 1-5 stars
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship to user
    user = db.relationship('User', backref='feedbacks')

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route("/")
def home():
    return jsonify({'message': 'ChefGPT API is running'})

@app.route('/register', methods=["POST"])  # remove GET, Flutter sends forms as JSON
def register():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, email=email, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Registration successful!'})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")

    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        login_user(user)
        return jsonify({'success': True, 'username': user.username, 'user_id': user.id})
    return jsonify({'error': 'Invalid username or password'}), 401

@app.route("/dashboard")
@login_required
def dashboard():
    recommender = get_recommender()
    df = recommender.df.copy()
    df = df.drop_duplicates(subset=['recipe_id'], keep='first')
    df = df.drop_duplicates(subset=['recipe_name'], keep='first')
    df_shuffled = df.sample(n=min(12, len(df)), replace=False)

    recipes = []
    for _, row in df_shuffled.iterrows():
        recipes.append({
            'recipe_id': int(row['recipe_id']),
            'recipe_name': str(row['recipe_name']),
            'ingredients': str(row['ingredients']),
            'cuisine': str(row['cuisine']),
            'calories': int(row['calories']),
            'rating': float(row['rating'])
        })

    return jsonify({'username': current_user.username, 'recipes': recipes, 'total_recipes': len(df)})

# Add this new route for Load More functionality in dashboard catalogue
@app.route('/load-more-recipes')
@login_required
def load_more_recipes():
    offset = request.args.get('offset', 0, type=int)
    limit = 12
    
    recommender = get_recommender()
    df = recommender.df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['recipe_name'], keep='first')
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Get recipes for this batch
    recipes_df = df.iloc[offset:offset + limit]
    
    recipes_list = []
    for _, row in recipes_df.iterrows():
        recipes_list.append({
            'recipe_id': int(row['recipe_id']),
            'recipe_name': str(row['recipe_name']),
            'ingredients': str(row['ingredients']),
            'cuisine': str(row['cuisine']),
            'calories': int(row['calories']),
            'rating': float(row['rating'])
        })
    
    has_more = (offset + limit) < len(df)
    
    return jsonify({
        'recipes': recipes_list,
        'has_more': has_more
    })


@app.route("/get-recommendations", methods=["POST"])
@login_required
def get_recommendations():
    try:
        data = request.get_json()
        user_ingredients = data.get('ingredients', '')
        
        if not user_ingredients:
            return jsonify({'error': 'Please enter at least one ingredient'}), 400
        
        # Get ML recommender
        recommender = get_recommender()
        
        # Get recommendations
        recommendations = recommender.recommend(user_ingredients, top_k=5)
        
        return jsonify({'recommendations': recommendations})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/start-cooking/<int:recipe_id>")
@login_required
def start_cooking(recipe_id):
    recommender = get_recommender()
    recipe = recommender.get_recipe_by_id(recipe_id)
    if not recipe:
        return jsonify({'error': 'Recipe not found'}), 404
    return jsonify({'recipe': recipe})

@app.route("/recipe/<int:recipe_id>")
@login_required
def recipe_detail(recipe_id):
    recommender = get_recommender()
    recipe = recommender.get_recipe_by_id(recipe_id)
    if not recipe:
        return jsonify({'error': 'Recipe not found'}), 404
    return jsonify({'recipe': recipe})

@app.route("/mark-as-cooked/<int:recipe_id>", methods=["POST"])
@login_required
def mark_as_cooked(recipe_id):
    try:
        recommender = get_recommender()
        recipe = recommender.get_recipe_by_id(recipe_id)
        
        if not recipe:
            return jsonify({'error': 'Recipe not found'}), 404
        
        # Check if already cooked today
        existing = CookedRecipe.query.filter_by(
            user_id=current_user.id,
            recipe_id=recipe_id
        ).first()
        
        # Add to cooked recipes
        cooked = CookedRecipe(
            user_id=current_user.id,
            recipe_id=recipe_id,
            recipe_name=recipe['recipe_name']
        )
        db.session.add(cooked)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'Recipe marked as cooked!'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/cooked-history")
@login_required
def cooked_history():
    cooked_recipes = CookedRecipe.query.filter_by(user_id=current_user.id)\
        .order_by(CookedRecipe.cooked_at.desc()).all()
    result = [{'recipe_name': r.recipe_name, 'cooked_at': r.cooked_at.strftime('%B %d, %Y')} for r in cooked_recipes]
    return jsonify({'history': result})

@app.route("/submit-feedback/<int:recipe_id>", methods=["POST"])
@login_required
def submit_feedback(recipe_id):
    try:
        data = request.get_json()
        rating = data.get('rating')
        comment = data.get('comment', '').strip()
        
        recommender = get_recommender()
        recipe = recommender.get_recipe_by_id(recipe_id)
        
        if not recipe:
            return jsonify({'error': 'Recipe not found'}), 404
        
        # Check if user already gave feedback for this recipe
        existing_feedback = Feedback.query.filter_by(
            user_id=current_user.id,
            recipe_id=recipe_id
        ).first()
        
        if existing_feedback:
            # Update existing feedback
            existing_feedback.rating = rating
            existing_feedback.comment = comment
            existing_feedback.created_at = datetime.utcnow()
        else:
            # Create new feedback
            feedback = Feedback(
                user_id=current_user.id,
                recipe_id=recipe_id,
                recipe_name=recipe['recipe_name'],
                rating=rating,
                comment=comment
            )
            db.session.add(feedback)
        
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'message': 'Thank you for your feedback!'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/get-feedbacks/<int:recipe_id>")
@login_required
def get_feedbacks(recipe_id):
    try:
        feedbacks = Feedback.query.filter_by(recipe_id=recipe_id)\
            .order_by(Feedback.created_at.desc()).all()
        
        feedback_list = []
        for fb in feedbacks:
            feedback_list.append({
                'username': fb.user.username,
                'rating': fb.rating,
                'comment': fb.comment,
                'created_at': fb.created_at.strftime('%B %d, %Y')
            })
        
        return jsonify({'feedbacks': feedback_list})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/search-recipes', methods=['GET'])
@login_required
def search_recipes():
    query = request.args.get('q', '').strip().lower()
    
    if not query:
        return jsonify({'recipes': []})
    
    recommender = get_recommender()
    df = recommender.df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['recipe_name'], keep='first')
    
    # Filter recipes
    mask = (
        df['recipe_name'].str.lower().str.contains(query, na=False) |
        df['cuisine'].str.lower().str.contains(query, na=False) |
        df['ingredients'].str.lower().str.contains(query, na=False)
    )
    
    filtered_df = df[mask]
    
    # Convert to dict and fix data types
    recipes_list = []
    for _, row in filtered_df.iterrows():
        recipes_list.append({
            'recipe_id': int(row['recipe_id']),
            'recipe_name': str(row['recipe_name']),
            'ingredients': str(row['ingredients']),
            'cuisine': str(row['cuisine']),
            'calories': int(row['calories']),
            'rating': float(row['rating']),
            'difficulty': str(row['difficulty'])
        })
    
    return jsonify({'recipes': recipes_list})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)