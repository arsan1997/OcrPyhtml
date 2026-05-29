from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import easyocr
import base64
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

print('Loading EasyOCR...')
reader = easyocr.Reader(['en'])
print('EasyOCR Ready!')

TABLE_ROIS = [
    [0.292, 0.500, 0.335, 0.940],
    [0.562, 0.040, 0.605, 0.480],
    [0.562, 0.500, 0.605, 0.940],
    [0.838, 0.040, 0.881, 0.480],
    [0.838, 0.500, 0.881, 0.940]
]

@app.route('/ocr', methods=['POST'])
def process_ocr():
    try:
        data = request.json
        image_data = data.get('image')
        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400

        header, encoded = image_data.split(',', 1)
        decoded = base64.b64decode(encoded)
        img_pil = Image.open(io.BytesIO(decoded))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        h_img, w_img, _ = img.shape
        
        # --- Auto-Rotation Logic ---
        if w_img > h_img:
            print("Detect Landscape: Rotating to Portrait...")
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            h_img, w_img, _ = img.shape
        
        all_answers = []

        for i, roi in enumerate(TABLE_ROIS):
            ymin, xmin, ymax, xmax = int(roi[0]*h_img), int(roi[1]*w_img), int(roi[2]*h_img), int(roi[3]*w_img)
            table_row = img[ymin:ymax, xmin:xmax]
            h_row, w_row, _ = table_row.shape
            col_width = w_row / 10
            
            for col_idx in range(10):
                x_start = int(col_idx * col_width)
                x_end = int((col_idx + 1) * col_width)
                cell = table_row[2:h_row-2, x_start+3 : x_end-3]
                cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                cell_resized = cv2.resize(cell_gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                
                result = reader.readtext(
                    cell_resized, 
                    detail=0, 
                    decoder='greedy',
                    paragraph=False,
                    allowlist='0123456789',
                    text_threshold=0.2,
                    low_text=0.3,
                    canvas_size=1200
                )
                
                if result and result[0].strip().isdigit():
                    all_answers.append(result[0].strip())
                else:
                    cell_rgb = cv2.cvtColor(cell_resized, cv2.COLOR_GRAY2RGB)
                    alt_result = reader.readtext(cell_rgb, detail=0, decoder='greedy', allowlist='0123456789', text_threshold=0.2)
                    if alt_result and alt_result[0].strip().isdigit():
                        all_answers.append(alt_result[0].strip())
                    else:
                        all_answers.append('?')

        return jsonify({'answers': all_answers})

    except Exception as e:
        print(f'Error: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
