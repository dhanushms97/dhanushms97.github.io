from imutils import paths #imutils includes opencv functions
import face_recognition
import pickle
import cv2
import os
#get paths of each file in folder named Images
#Images here that contains data(folders of various people)
imagePath = list(paths.list_images('Images'))
kEncodings = []
kNames = []
# loop over the image paths
for (i, ip) in enumerate(imagePath):
# extract the person name from the image path
name = ip.split(os.path.sep)[-2]
# load the input image and convert it from BGR
image = cv2.imread(ip)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes = face_recognition.face_locations(rgb,model='hog')
# compute the facial embedding for the any face
encodings = face_recognition.face_encodings(rgb, boxes)
# loop over the encodings
for encoding in encodings:
kEncodings.append(encoding)
kNames.append(name)
#save emcodings along with their names in dictionary data
data = {"encodings": kEncodings, "names": kNames}
#use pickle to save data into a file for later use
f = open("face_enc", "wb")
f.write(pickle.dumps(data))#to open file in write mode
f.close()#to close file
Now see we have stored the embedding in a file named “face_enc“, so we can use them to recognize faces in images(photographs) or live video stream.
The next part is that we are going to see how to recognize faces from images.

How to Recognize Face in Images
The following script for detecting and recognizing faces in images. I have given comments next to code wherever it is required. So You will able to understand code efficiently.

import face_recognition
import imutils #imutils includes opencv functions
import pickle
import time
import cv2
import os
#to find path of xml file containing haarCascade file
cfp = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
fc = cv2.CascadeClassifier(cfp)
# load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read())
#Find path to the image you want to detect face and pass it here
image = cv2.imread(Path-to-img)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#convert image to Greyscale for HaarCascade
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = fc.detectMultiScale(gray,
scaleFactor=1.1,
minNeighbors=6,
minSize=(60, 60),
flags=cv2.CASCADE_SCALE_IMAGE)
# the facial embeddings for face in input
encodings = face_recognition.face_encodings(rgb)
names = []
# loop over the facial embeddings incase
# we have multiple embeddings for multiple fcaes
for encoding in encodings:
#Compare encodings with encodings in data["encodings"]
#Matches contain array with boolean values True and False
matches = face_recognition.compare_faces(data["encodings"],
encoding)
#set name =unknown if no encoding matches
name = "Unknown"
# check to see if we have found a match
if True in matches:
#Find positions at which we get True and store them
matchedIdxs = [i for (i, b) in enumerate(matches) if b]
count = {}
# loop over the matched indexes and maintain a count for
# each recognized face face
for i in matchedIdxs:
#Check the names at respective indexes we stored in matchedIdxs
name = data["names"][i]
#increase count for the name we got
count[name] = count.get(name, 0) + 1
#set name which has highest count
name = max(count, key=count.get)
# will update the list of names
names.append(name)
for ((x, y, w, h), name) in zip(faces, names):
cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
0.75, (0, 255, 0), 2)
cv2.imshow("Frame", image)
cv2.waitKey(0)