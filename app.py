import os
import base64
import re
import sqlite3
from io import BytesIO
from flask import Flask, render_template, request, g, jsonify
from PIL import Image
import fitz  # PyMuPDF
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
DB = 'database.db'
MODEL_PATH = r"C:\Users\Abhay Pandey\Desktop\pdf_app\models\best (3).pt"
model = YOLO(MODEL_PATH)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB)
    return g.db

@app.teardown_appcontext
def close_db(error):
    db = g.pop('db', None)
    if db:
        db.close()

def init_db():
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS extracted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_name TEXT,
            page INTEGER,
            item_name TEXT,
            in_date TEXT,
            in_time TEXT,
            vendor_name TEXT,
            po_number TEXT
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS page_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            data_id INTEGER,
            image_base64 TEXT,
            FOREIGN KEY (data_id) REFERENCES extracted_data(id)
        )
    ''')
    db.commit()

def extract_fields_from_text(text):
    patterns = {
        "Item Name": r'Item Name\s*[:\-]?\s*([^\n\r:]+)',
        "In Date": r'In Date\s*[:\-]?\s*([\d]{2,4}[-/][\d]{1,2}[-/][\d]{1,4})',
        "In Time": r'In\s*time\s*[:\-]?\s*([\d]{1,2}:[\d]{2}(?::[\d]{2})?)',
        "Vendor Name": r'Vendor Name\s*[:\-]?\s*([^\n\r:]+)',
        "PO Number": r'PO\s*No\s*[:\-#]?\s*([A-Za-z0-9\-\/]+)'
    }

    fields = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        fields[key] = match.group(1).strip() if match else None
    return fields

@app.route('/', methods=['GET', 'POST'])
def index():
    init_db()
    db = get_db()

    # Read plant list from file
    with open('Plants lists.txt', 'r', encoding='utf-8') as f:
        plants = [line.strip() for line in f if line.strip()]

    if request.method == 'POST':
        file = request.files.get('pdfs')
        if not file:
            return render_template('index.html', rows=[], plants=plants)

        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        doc = fitz.open(filepath)
        for page_number, page in enumerate(doc):
            text = page.get_text()
            fields = extract_fields_from_text(text)

            if not any(fields.values()):
                continue

            cur = db.execute('''
                INSERT INTO extracted_data 
                (pdf_name, page, item_name, in_date, in_time, vendor_name, po_number)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                filename, page_number + 1,
                fields.get("Item Name"),
                fields.get("In Date"),
                fields.get("In Time"),
                fields.get("Vendor Name"),
                fields.get("PO Number")
            ))
            data_id = cur.lastrowid

            images = page.get_images(full=True)
            for img in images:
                xref = img[0]
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
                pil_img.thumbnail((120, 120))
                thumb_io = BytesIO()
                pil_img.save(thumb_io, format="PNG")
                image_data = base64.b64encode(thumb_io.getvalue()).decode()

                db.execute('''
                    INSERT INTO page_images (data_id, image_base64)
                    VALUES (?, ?)
                ''', (data_id, image_data))

            db.commit()
        doc.close()

    rows = db.execute('SELECT * FROM extracted_data ORDER BY id DESC').fetchall()
    data_with_images = []
    for row in rows:
        images = db.execute('SELECT image_base64 FROM page_images WHERE data_id = ?', (row[0],)).fetchall()
        data_with_images.append({
            "row": row,
            "images": [img[0] for img in images]
        })

    return render_template('index.html', rows=data_with_images, plants=plants)

@app.route('/reset', methods=['POST'])
def reset():
    db = get_db()
    db.execute('DELETE FROM page_images')
    db.execute('DELETE FROM extracted_data')
    db.commit()

    for f in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, f))

    return '', 204

@app.route('/detect', methods=['POST'])
def detect():
    image_base64 = request.form.get('image')
    if not image_base64:
        return jsonify({"labels": []})

    img_bytes = BytesIO(base64.b64decode(image_base64))
    img = Image.open(img_bytes).convert("RGB")
    results = model.predict(np.array(img), verbose=False)
    labels = list(set([model.names[int(box.cls)] for box in results[0].boxes]))

    return jsonify({"labels": labels})
if __name__ == '__main__':
    app.run(debug=True)
