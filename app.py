import os
import io
import base64
import sqlite3
import functools
import numpy as np
import click

from flask import (
    Flask, request, render_template, redirect,
    url_for, session, g, flash
)
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image
from skimage import color, filters, measure, morphology
from skimage.morphology import square

# ——— Configuration ———
database_path = os.path.join(os.path.dirname(__file__), 'users.db')
SECRET_KEY = os.environ.get('SECRET_KEY') or 'your_secret_key_here'

app = Flask(__name__)
app.config.from_mapping(
    SECRET_KEY=SECRET_KEY,
    DATABASE=database_path
)

##############################
# Database helper functions  #
##############################

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    db = get_db()
    with app.open_resource('schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

@app.cli.command('init-db')
def init_db_command():
    """Clear existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')

#########################
# Authentication helper #
#########################

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return view(**kwargs)
    return wrapped_view

########################
# Authentication routes#
########################
from datetime import datetime, date

@app.route('/dashboard')
@login_required
def dashboard():
    db = get_db()

    # Get all unique towns for the dropdown
    towns = db.execute(
        'SELECT DISTINCT location FROM analysis WHERE user_id = ?', (session['user_id'],)
    ).fetchall()
    town_list = [row['location'] for row in towns]

    # Get selected town from query string
    selected_town = request.args.get('town', '')

    # Filter by town if selected
    params = [session['user_id']]
    query = '''
        SELECT capture_date,
               AVG(area) AS avg_area,
               AVG(perimeter) AS avg_perimeter,
               AVG(eccentricity) AS avg_eccentricity,
               AVG(solidity) AS avg_solidity
        FROM analysis
        WHERE user_id = ?
    '''
    if selected_town:
        query += ' AND location = ?'
        params.append(selected_town)
    query += '''
        GROUP BY capture_date
        ORDER BY capture_date
    '''
    analyses = db.execute(query, params).fetchall()

    # Format dates and features
    from datetime import datetime, date
    dates = []
    for row in analyses:
        d = row['capture_date']
        if isinstance(d, (datetime, date)):
            formatted_date = d.strftime("%m/%d/%Y")
        else:
            formatted_date = datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%Y")
        formatted_date = '/'.join([str(int(part)) for part in formatted_date.split('/')])
        dates.append(formatted_date)

    area         = [row['avg_area']        for row in analyses]
    perimeter    = [row['avg_perimeter']   for row in analyses]
    eccentricity = [row['avg_eccentricity'] for row in analyses]
    solidity     = [row['avg_solidity']     for row in analyses]

    return render_template(
        'dashboard.html',
        dates=dates,
        area=area,
        perimeter=perimeter,
        eccentricity=eccentricity,
        solidity=solidity,
        towns=town_list,
        selected_town=selected_town
    )

@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        db = get_db()
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'
        else:
            # You can still check first, but we'll also catch the exception below.
            existing = db.execute(
                'SELECT id FROM user WHERE username = ?', (username,)
            ).fetchone()
            if existing:
                error = f"Username “{username}” is already taken."

        if error is None:
            try:
                db.execute(
                    'INSERT INTO user (username, password) VALUES (?, ?)',
                    (username, generate_password_hash(password))
                )
                db.commit()
                #flash('Registration successful! Please log in.')
                return redirect(url_for('login'))
            except sqlite3.IntegrityError:
                # This will catch the UNIQUE constraint failure if it slips through
                error = f"Username “{username}” is already taken."

        flash(error)

    return render_template('register.html')

@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None

        user = db.execute(
            'SELECT * FROM user WHERE username = ?', (username,)
        ).fetchone()

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            #flash('Logged in successfully.')
            return redirect(url_for('index'))

        flash(error)

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('index'))

########################
# Image analysis utils #
########################

def preprocess_image(image, target_size=(256, 256)):
    image = image.resize(target_size)
    image_np = np.array(image) / 255.0
    return np.expand_dims(image_np, axis=0)

def postprocess_prediction(prediction):
    gray = color.rgb2gray(prediction)
    try:
        thresh = filters.threshold_otsu(gray)
    except Exception:
        thresh = 0.5
    mask = gray > thresh
    closed = morphology.closing(mask, square(3))
    dilated = morphology.dilation(closed, square(3))
    return morphology.remove_small_objects(dilated, min_size=50)

def extract_features(mask):
    labeled = measure.label(mask)
    regions = measure.regionprops(labeled)
    if not regions:
        return None
    r = max(regions, key=lambda reg: reg.area)
    return {
        'Area (px²)': r.area,
        'Perimeter (px)': r.perimeter,
        'Eccentricity': r.eccentricity,
        'Solidity': r.solidity,
        'Centroid Row': r.centroid[0],
        'Centroid Col': r.centroid[1]
    }

def image_to_base64(arr, mode="L"):
    im = Image.fromarray((arr * 255).astype(np.uint8), mode=mode)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def is_coffee_bean(img, bean_model, img_size=(224, 224)):
    img_resized = img.resize(img_size)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = bean_model.predict(img_array)
    return "coffee bean" if pred[0][0] < 0.5 else "not coffee bean"

########################
# Main application routes
########################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze')
@login_required
def analyze():
    return render_template('analyze.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    import tensorflow as tf  # Lazy import
    MODEL_PATH = 'trained_autoencoder.h5'
    BEAN_MODEL_PATH = 'beanVer1.h5'

    # Load models only when needed
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    )
    bean_model = tf.keras.models.load_model(BEAN_MODEL_PATH)

    files = request.files.getlist('file')
    if not files or files[0].filename == '':
        return "No file provided", 400

    results = []
    for file in files:
        try:
            img = Image.open(file.stream).convert("RGB")
            proc = preprocess_image(img)
            pred = model.predict(proc)[0]
            mask = postprocess_prediction(pred)
            feats = extract_features(mask)
            if feats is None:
                continue  # Skip images with no bean detected

            orig_b64 = image_to_base64(np.array(img) / 255.0, mode="RGB")
            mask_b64 = image_to_base64(mask.astype(np.float32), mode="L")

            bean_label = is_coffee_bean(img, bean_model)  # Pass bean_model

            results.append({
                'features': feats,
                'original_image': orig_b64,
                'segmentation_image': mask_b64,
                'bean_label': bean_label
            })
        except Exception as e:
            continue

    if not results:
        return "No valid beans detected in any image.", 400

    return render_template('result.html', results=results)

def safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0

@app.route('/save_result', methods=['POST'])
@login_required
def save_result():
    db = get_db()
    # Find all indices for images
    indices = []
    for key in request.form.keys():
        if key.startswith('feature_names_'):
            indices.append(key.split('_')[2].split('[')[0])
    indices = sorted(set(indices), key=int)

    loc = request.form.get('location')
    ctype = request.form.get('coffee_type')
    cdate = request.form.get('capture_date')
    remarks = request.form.get('remarks')

    try:
        for idx in indices:
            feature_names = request.form.getlist(f'feature_names_{idx}[]')
            feature_values = request.form.getlist(f'feature_values_{idx}[]')
            feats = dict(zip(feature_names, feature_values))

            db.execute(
                '''INSERT INTO analysis
                   (user_id, location, coffee_type, capture_date, remarks,
                    area, perimeter, eccentricity, solidity,
                    centroid_row, centroid_col)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    session['user_id'], loc, ctype, cdate, remarks,
                    safe_float(feats.get('Area (px²)', 0)),
                    safe_float(feats.get('Perimeter (px)', 0)),
                    safe_float(feats.get('Eccentricity', 0)),
                    safe_float(feats.get('Solidity', 0)),
                    safe_float(feats.get('Centroid Row', 0)),
                    safe_float(feats.get('Centroid Col', 0))
                )
            )
        db.commit()
        flash('All analyses saved successfully!', 'success')
    except Exception as e:
        db.rollback()
        flash(f'Error saving analyses: {e}')

    return redirect(url_for('analyze'))

from flask import Response
import csv

@app.route('/download_data')
def download_data():
    db = get_db()
    data = db.execute(
        '''SELECT location, coffee_type, area, perimeter, eccentricity, solidity
           FROM analysis'''
    ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    # CSV header
    writer.writerow(['Location', 'Coffee Type', 'Area (px²)', 'Perimeter (px)', 'Eccentricity', 'Solidity'])

    for row in data:
        writer.writerow([row['location'], row['coffee_type'], row['area'], row['perimeter'], row['eccentricity'], row['solidity']])

    response = Response(output.getvalue(), mimetype='text/csv')
    response.headers['Content-Disposition'] = 'attachment; filename=coffee_data.csv'
    return response

#########################
# Data management route #
#########################
@app.route('/data')
@login_required
def data():
    db = get_db()
    query = 'SELECT * FROM analysis WHERE 1=1'
    params = []

    location = request.args.get('location', '').strip()
    coffee_type = request.args.get('coffee_type', '').strip()
    capture_date = request.args.get('capture_date', '').strip()
    remarks = request.args.get('remarks', '').strip()
    page = int(request.args.get('page', 1))
    per_page = 50

    if location:
        query += ' AND location LIKE ?'
        params.append(f'%{location}%')
    if coffee_type:
        query += ' AND coffee_type LIKE ?'
        params.append(f'%{coffee_type}%')
    if capture_date:
        query += ' AND capture_date = ?'
        params.append(capture_date)
    if remarks:
        query += ' AND remarks = ?'
        params.append(remarks)

    # Get total count for pagination
    count_query = 'SELECT COUNT(*) FROM (' + query + ')'
    total = db.execute(count_query, params).fetchone()[0]
    pages = (total + per_page - 1) // per_page

    # Add LIMIT and OFFSET for pagination
    query += ' ORDER BY id ASC LIMIT ? OFFSET ?'
    params.extend([per_page, (page - 1) * per_page])

    analyses = db.execute(query, params).fetchall()
    return render_template('data.html', analyses=analyses, page=page, pages=pages, request=request)

@app.route('/edit_analysis/<int:analysis_id>', methods=['POST'])
@login_required
def edit_analysis(analysis_id):
    db = get_db()
    db.execute('''
        UPDATE analysis SET
            location = ?,
            coffee_type = ?,
            capture_date = ?,
            remarks = ?,
            area = ?,
            perimeter = ?,
            eccentricity = ?,
            solidity = ?,
            centroid_row = ?,
            centroid_col = ?
        WHERE id = ?
    ''', (
        request.form['location'],
        request.form['coffee_type'],
        request.form['capture_date'],
        request.form['remarks'],
        request.form['area'],
        request.form['perimeter'],
        request.form['eccentricity'],
        request.form['solidity'],
        request.form['centroid_row'],
        request.form['centroid_col'],
        analysis_id
    ))
    db.commit()
#    flash('Analysis updated.')
    return redirect(url_for('data'))

@app.route('/delete_analysis/<int:analysis_id>')
@login_required
def delete_analysis(analysis_id):
    db = get_db()
    db.execute('DELETE FROM analysis WHERE id = ?', (analysis_id,))
    db.commit()
#    flash('Analysis deleted.')
    return redirect(url_for('data'))

###################
# Reporting route #
###################
@app.route('/reports')
def reports():
    db = get_db()
    user_id = session.get('user_id')

    # Get all unique towns and coffee types for the dropdowns
    if user_id:
        towns = db.execute(
            'SELECT DISTINCT location FROM analysis'  # removed WHERE user_id = ?
        ).fetchall()
        coffee_types = db.execute(
            'SELECT DISTINCT coffee_type FROM analysis'
        ).fetchall()
        remarks_list = db.execute(
            'SELECT DISTINCT remarks FROM analysis'
        ).fetchall()
    else:
        towns = db.execute(
            'SELECT DISTINCT location FROM analysis'
        ).fetchall()
        coffee_types = db.execute(
            'SELECT DISTINCT coffee_type FROM analysis'
        ).fetchall()
        remarks_list = db.execute(
            'SELECT DISTINCT remarks FROM analysis'
        ).fetchall()

    town_list = [row['location'] for row in towns]
    coffee_type_list = [row['coffee_type'] for row in coffee_types]
    remarks = [row['remarks'] for row in remarks_list]

    # Get filters from query string
    selected_town = request.args.get('town', '')
    selected_coffee_type = request.args.get('coffee_type', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    remarks_filter = request.args.get('remarks', '')

    # Build query with filters
    params = []
    query = 'SELECT * FROM analysis WHERE 1=1'
    # REMOVE THIS BLOCK:
    # if user_id:
    #     query += ' AND user_id = ?'
    #     params.append(user_id)
    if selected_town:
        query += ' AND location = ?'
        params.append(selected_town)
    if selected_coffee_type:
        query += ' AND coffee_type = ?'
        params.append(selected_coffee_type)
    if remarks_filter:
        query += ' AND remarks = ?'
        params.append(remarks_filter)
    if start_date:
        query += ' AND capture_date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND capture_date <= ?'
        params.append(end_date)
    query += ' ORDER BY capture_date'

    all_data = db.execute(query, params).fetchall()

    # Prepare data for charts (group by date)
    from collections import defaultdict
    from datetime import datetime, date

    chart_data = defaultdict(lambda: {'area': [], 'perimeter': [], 'eccentricity': [], 'solidity': []})
    for row in all_data:
        d = row['capture_date']
        if isinstance(d, (datetime, date)):
            formatted_date = d.strftime("%m/%d/%Y")
        else:
            formatted_date = datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d/%Y")
        formatted_date = '/'.join([str(int(part)) for part in formatted_date.split('/')])
        chart_data[formatted_date]['area'].append(row['area'])
        chart_data[formatted_date]['perimeter'].append(row['perimeter'])
        chart_data[formatted_date]['eccentricity'].append(row['eccentricity'])
        chart_data[formatted_date]['solidity'].append(row['solidity'])

    dates = sorted(chart_data.keys(), key=lambda x: datetime.strptime(x, "%m/%d/%Y"))
    area = [sum(chart_data[d]['area'])/len(chart_data[d]['area']) if chart_data[d]['area'] else 0 for d in dates]
    perimeter = [sum(chart_data[d]['perimeter'])/len(chart_data[d]['perimeter']) if chart_data[d]['perimeter'] else 0 for d in dates]
    eccentricity = [sum(chart_data[d]['eccentricity'])/len(chart_data[d]['eccentricity']) if chart_data[d]['eccentricity'] else 0 for d in dates]
    solidity = [sum(chart_data[d]['solidity'])/len(chart_data[d]['solidity']) if chart_data[d]['solidity'] else 0 for d in dates]

    return render_template(
        'reports.html',
        dates=dates,
        remarks=remarks,
        area=area,
        perimeter=perimeter,
        eccentricity=eccentricity,
        solidity=solidity,
        towns=town_list,
        selected_town=selected_town,
        coffee_types=coffee_type_list,
        selected_coffee_type=selected_coffee_type,
        start_date=start_date,
        end_date=end_date,
        all_data=all_data
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
