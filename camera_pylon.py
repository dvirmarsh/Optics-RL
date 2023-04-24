import pypylon.pylon as py
from PIL import Image


tlf = py.TlFactory.GetInstance()

cam = py.InstantCamera(tlf.CreateFirstDevice())
cam.Open()

#cam.ExposureTime.From = 10 # in microseconds
#cam.ExposureTime.To = 50 # in microseconds
#cam.ExposureTime.Increment = 10

cam.StartGrabbingMax(5)
i=0
while cam.IsGrabbing():
    i+=1
    with cam.RetrieveResult(1000) as res:
        if res.GrabSucceeded():
            img = res.Array
            im = Image.fromarray(img)
            im.save("image"+str(i)+".jpeg")
        else:
            print("error")



a = 0
cam.Close()