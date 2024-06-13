Project Overview

This project is a web-based application built with Flask, OpenCV, and the face_recognition library. It provides real-time face recognition using a webcam, along with functionalities to upload, capture, and manage face images. The application allows users to recognize known faces, add new faces to the database, and delete existing ones.

Features

- Real-time Face Recognition: Streams video from the user's webcam and recognizes faces in real-time.

- Upload Face Images: Allows users to upload images of new faces to be added to the recognition database.

- Capture Face Images: Provides a feature to capture face images directly from the webcam and add them to the database.

- Manage Face Images: Users can view and delete face images from the database.

- View Recognized Faces: Displays the names of recognized faces from the webcam feed.

Technologies Used

- Flask: A lightweight WSGI web application framework in Python.

- OpenCV: An open-source computer vision and machine learning software library.

- face_recognition: A simple and easy-to-use library for face recognition in Python.

- HTML/CSS/JavaScript: For creating the web pages and user interface.

- SQLite: For storing face encodings and names persistently.


Project Structure

face_recognition_flask_app/

app.py
->templates/
    index.html
    index1.html
    capture.html
    upload.html
    manage_photos.html
    
->static/
    css/
    js/
    
->uploads/
    (uploaded face images)

->encodings.pkl
->names.pkl
->requirements.txt

