from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import base64
from PIL import Image
import io

app = Flask(__name__)

# Global variables
model = load_model('best_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
camera = None
is_streaming = False

def process_face(face):
    face = cv2.resize(face, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=[0, -1])
    prediction = model.predict(face)
    return emotions[np.argmax(prediction[0])]

def generate_frames():
    global camera, is_streaming
    camera = cv2.VideoCapture(0)
    
    while is_streaming:
        success, frame = camera.read()
        if not success:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            emotion = process_face(face_roi)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_stream():
    global is_streaming
    is_streaming = True
    return jsonify({"status": "started"})

@app.route('/stop')
def stop_stream():
    global is_streaming, camera
    is_streaming = False
    if camera:
        camera.release()
    return jsonify({"status": "stopped"})

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Get image from request
        image_data = request.json['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to opencv format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        results = []
        emotions_text = []  # Store emotions for display
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            emotion = process_face(face_roi)
            emotions_text.append(emotion)  # Add emotion to list
            results.append({
                "emotion": emotion,
                "position": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
            })
            
            # Draw on image
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, emotion, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert processed image back to base64
        _, buffer = cv2.imencode('.jpg', image)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "status": "success",
            "results": results,
            "emotions": emotions_text,  # Send emotions list
            "processed_image": f"data:image/jpeg;base64,{processed_image}"
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True) 