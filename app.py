from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from video_processor import VideoProcessor

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Cấu hình upload
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Khởi tạo VideoProcessor
video_processor = VideoProcessor('model_cnn10.pt')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file')
            return redirect(request.url)
        
        file = request.files['video']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            result_path = os.path.join(app.config['RESULT_FOLDER'], f'processed_{filename}')
            
            # Save uploaded video
            file.save(upload_path)
            
            # Process video
            try:
                behavior, confidence = video_processor.process_video(upload_path, result_path)
                
                # Construct paths for template
                original_video_path = f'uploads/{filename}'
                processed_video_path = f'results/processed_{filename}'
                
                # Debugging paths
                print(f"Original video path: {original_video_path}")
                print(f"Processed video path: {processed_video_path}")
                
                return render_template('result10.html', 
                                       original_video=original_video_path,  
                                       processed_video=processed_video_path,
                                       behavior=behavior,
                                       confidence=confidence)
            except Exception as e:
                flash(f'Error processing video: {str(e)}')
                return redirect(request.url)
    return render_template('index10.html')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)