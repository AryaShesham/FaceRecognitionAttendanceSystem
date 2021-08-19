import datetime
import os
from datetime import datetime
import cv2
import face_recognition
import matplotlib as mpl
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.shortcuts import render, redirect
from .forms import usernameForm
mpl.use('Agg')


# utility functions:
def username_present(username):
    if User.objects.filter(username=username).exists():
        return True

    return False


def create_dataset(username):
    name = username
    if not os.path.exists('face_recognition_data/training_dataset/{}/'.format(name)):
        os.makedirs('face_recognition_data/training_dataset/{}/'.format(name))
    directory = 'face_recognition_data/training_dataset/{}/'.format(name)

    if not os.path.exists(f"attendance_data/{name}/"):
        os.makedirs(f"attendance_data/{name}/")
        with open(f'attendance_data/{name}/Attendance.csv', 'x') as f:
            f.write('Date, Username, Login Time, Logout Time:\n')

    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Taking Images...")

    img_counter = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab images..")
            break

        cv2.imshow("Click Space-Bar to take images", frame)

        k = cv2.waitKey(1)

        if k % 256 == 27:  # Pressed Esc Key
            print("Escape pressed: Closing the app")
            break

        elif k % 256 == 32:  # Pressed Space-Bar
            img_name = f"{name}_{img_counter}.jpg"
            cv2.imwrite(directory + '/' + img_name, frame)
            print("Image Taken...")
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


def total_number_employees():
    qs = User.objects.all()
    return (len(qs) - 1)



# Create your views here.
def home(request):
    return render(request, 'recognition/home.html')


@login_required
def dashboard(request):
    if request.user.username == 'admin':
        print("admin")
        return render(request, 'recognition/admin_dashboard.html')
    else:
        print("not admin")

        return render(request, 'recognition/employee_dashboard.html')


@login_required
def add_photos(request):
    if request.user.username != 'admin':
        return redirect('not-authorised')
    if request.method == 'POST':
        form = usernameForm(request.POST)
        data = request.POST.copy()
        username = data.get('username')
        if username_present(username):
            create_dataset(username)
            messages.success(request, f'Dataset Created')
            return redirect('add-photos')
        else:
            messages.warning(request, f'No such username found. Please register employee first.')
            return redirect('dashboard')

    else:

        form = usernameForm()
        return render(request, 'recognition/add_photos.html', {'form': form})


@login_required
def mark_your_attendance(request):
    if request.user.username == 'admin':
        return render(request, 'recognition/home.html')

    else:
        imagesList = []
        path = f'face_recognition_data/training_dataset/{request.user.username}/'
        myList = os.listdir(path)

        for img in myList:
            currentImg = cv2.imread(f'{path}{img}')
            imagesList.append(currentImg)

        def do_encodings(images):
            encodeList = []
            for image in images:
                imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(imgRGB)[0]
                encodeList.append(encode)
            return encodeList

        encodeListForKnownFaces = do_encodings(imagesList)

        capture = cv2.VideoCapture(0)
        cv2.waitKey(1)

        while True:
            success, img = capture.read()
            cv2.imshow('capture', img)
            imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

            facesInCurrentFrame = face_recognition.face_locations(imgSmall)
            encodesOfCurrentFrame = face_recognition.face_encodings(imgSmall, facesInCurrentFrame)

            for faceLocation, encodingOfFace in zip(facesInCurrentFrame, encodesOfCurrentFrame):
                matches = face_recognition.compare_faces(encodeListForKnownFaces, encodingOfFace)
                faceDist = face_recognition.face_distance(encodeListForKnownFaces, encodingOfFace)

                for elem in faceDist:
                    cv2.waitKey(21)
                    if elem <= 0.45:

                        with open(f'attendance_data/{request.user.username}/Attendance.csv', 'r+') as f:

                            dataList = f.readlines()
                            attendanceNameList = []
                            for line in dataList:
                                entry = line.split(',')  # To split the line based on comma.
                                attendanceNameList.append(entry[0])

                            today = datetime.today()
                            today_date = today.strftime('%d-%b-%Y')
                            if today_date not in attendanceNameList:
                                now = datetime.now()
                                dateTimeString = now.strftime('%H:%M:%S')
                                f.writelines(f'\n{today_date}, {request.user.username}, log in: {dateTimeString}')

                            else:
                                print("Already Logged in today.")

                        return render(request, 'recognition/employee_dashboard.html')

                    else:
                        return render(request, 'recognition/home.html')
        capture.release()
        cv2.destroyAllWindows()


def mark_your_attendance_out(request):

    if request.user.username == 'admin':
        return render(request, 'recognition/home.html')

    else:
        imagesList = []
        path = f'face_recognition_data/training_dataset/{request.user.username}/'
        myList = os.listdir(path)

        for img in myList:
            currentImg = cv2.imread(f'{path}{img}')
            imagesList.append(currentImg)

        def do_encodings(images):
            encodeList = []
            for image in images:
                imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(imgRGB)[0]
                encodeList.append(encode)
            return encodeList

        encodeListForKnownFaces = do_encodings(imagesList)

        capture = cv2.VideoCapture(0)
        cv2.waitKey(1)

        while True:
            success, img = capture.read()
            cv2.imshow('capture', img)
            imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

            facesInCurrentFrame = face_recognition.face_locations(imgSmall)
            encodesOfCurrentFrame = face_recognition.face_encodings(imgSmall, facesInCurrentFrame)

            for faceLocation, encodingOfFace in zip(facesInCurrentFrame, encodesOfCurrentFrame):
                matches = face_recognition.compare_faces(encodeListForKnownFaces, encodingOfFace)
                faceDist = face_recognition.face_distance(encodeListForKnownFaces, encodingOfFace)

                for elem in faceDist:
                    cv2.waitKey(21)
                    if elem <= 0.45:

                        with open(f'attendance_data/{request.user.username}/Attendance.csv', 'r+') as f:

                            dataList = f.readlines()
                            attendanceNameList = []
                            for line in dataList:
                                entry = line.split(',')  # To split the line based on comma.
                                attendanceNameList.append(entry[0])

                            today = datetime.today()
                            today_date = today.strftime('%d-%b-%Y')

                            if today_date in attendanceNameList:
                                now = datetime.now()
                                dateTimeString = now.strftime('%H:%M:%S')
                                f.writelines(f', log out: {dateTimeString}.\n')

                            else:
                                print("Haven't logged in today")

                        return render(request, 'recognition/employee_dashboard.html')

                    else:
                        return render(request, 'recognition/home.html')

        capture.release()
        cv2.destroyAllWindows()


@login_required
def not_authorised(request):
    return render(request, 'recognition/not_authorised.html')