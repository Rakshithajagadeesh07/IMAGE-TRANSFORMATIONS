# IMAGE-TRANSFORMATIONS

## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import the required packages.

### Step2:
Load the image file in the program.

### Step3:
Use the techniques for Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

### Step4:
Display the modified image output

### Step5:
End the program.

## Program:
```python
Developed By : Rakshitha J
Register Number : 212223240135

i)Image Translation

import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image=cv2.imread("cat1.jpg")
input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print("Rakshitha J \n212223240135")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
M=np.float32([[1,0,100],[0,1,200],[0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(cols,rows))
plt.subplot(1, 2, 2)
plt.imshow(translated_image)
plt.axis('off')
plt.title("Image Translation")
plt.show()

ii) Image Scaling

input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print("Rakshitha J \n212223240135")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
M=np.float32([[1.5,0,0],[0,1.8,0],[0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(cols*2,rows*2))
plt.subplot(1, 2, 2)
plt.imshow(translated_image)
plt.axis('off')
plt.title("Image Scaling")
plt.show()

iii)Image shearing

input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print(" Rakshitha J\n212223240135")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
M1=np.float32([[1,0.5,0],[0,1,0],[0,0,1]])
M2=np.float32([[1,0,0],[0.5,1,0],[0,0,1]])
translated_image1=cv2.warpPerspective(input_image,M1,(int(cols*1.5),int(rows*1.5)))
translated_image2=cv2.warpPerspective(input_image,M2,(int(cols*1.5),int(rows*1.5)))
plt.subplot(1, 2, 2)
plt.imshow(translated_image1)
plt.imshow(translated_image2)
plt.axis('off')
plt.title("Image Shearing")
plt.show()

iv)Image Reflection

input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print("Rakshitha J \n212223240135")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
M1=np.float32([[1,0,0],[0,-1,rows],[0,0,1]])
M2=np.float32([[-1,0,cols],[0,1,0],[0,0,1]])
translated_image1=cv2.warpPerspective(input_image,M1,(int(cols),int(rows)))
translated_image2=cv2.warpPerspective(input_image,M2,(int(cols),int(rows)))
plt.subplot(1, 2, 2)
plt.imshow(translated_image1)
plt.imshow(translated_image2)
plt.axis('off')
plt.title("Image Reflection")
plt.show()

v)Image Rotation

input_image=cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)
print("Rakshitha J \n212223240135")
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
rows,cols,dim=input_image.shape
angle=np.radians(10)
M=np.float32([[np.cos(angle),-(np.sin(angle)),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
translated_image=cv2.warpPerspective(input_image,M,(int(cols),int(rows)))
plt.subplot(1, 2, 2)
plt.imshow(translated_image)
plt.axis('off')
plt.title("Image Rotation")
plt.show()

vi)Image Cropping

h, w, _ = input_image.shape
cropped_face = input_image[int(h*0.2):int(h*0.8), int(w*0.3):int(w*0.7)]
cv2.imwrite("cropped_face.png", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
print("Rakshitha J \n212223240135")
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title("Input Image")
plt.subplot(1, 2, 2)
plt.imshow(cropped_face)  
plt.axis('off')
plt.title("Cropped Image")
plt.show()

```
## Output:
### i)Image Translation

![Screenshot 2025-03-30 120241](https://github.com/user-attachments/assets/65fc7b1e-d893-427a-96b2-c354aa1b3d4d)

### ii) Image Scaling

![Screenshot 2025-03-30 120308](https://github.com/user-attachments/assets/4c58be69-b6bc-4873-bf3c-8bda3dd40a22)

### iii)Image shearing

![Screenshot 2025-03-30 120342](https://github.com/user-attachments/assets/cbc601d0-a0f5-4414-9fed-6b51a6d29628)

### iv)Image Reflection

![Screenshot 2025-03-30 120408](https://github.com/user-attachments/assets/659dd226-8c65-46bb-9433-51bb115146d8)

### v)Image Rotation

![Screenshot 2025-03-30 120440](https://github.com/user-attachments/assets/0d94578d-69da-42b7-977c-718f7077761e)

### vi)Image Cropping

![Screenshot 2025-03-30 120508](https://github.com/user-attachments/assets/eaa03287-df36-424f-8a11-e7345ff872e6)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
