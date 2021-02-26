from django.shortcuts import render
from .forms import ImageUploadForm
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import shutil



def handle_uploaded_image(f):
    with open('img.jpg', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    og=r'img.jpg'
    tg=r'media/img.jpg'
    shutil.move(og,tg)


# Create your views here.
def home(request):
    return render(request,'home.html')

def imageprocess(request):
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        handle_uploaded_image(request.FILES['image']) #input image
        model=ResNet50(weights='imagenet') #loading pre-trained model
        img_path = 'media/img.jpg' 
        #processing image to array (for easier/faster processing)
        img = image.load_img(img_path, target_size=(224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #prediction
        preds = model.predict(x)
        print('Predicted:', decode_predictions(preds, top=3)[0]) #testing in console
        #loading top 3 predictions into a array for sending it into html file 
        html = decode_predictions(preds, top=3)[0]
        res=[]
        for e in html:
            res.append((e[1],np.round(e[2]*100,2)))
        
        return render(request,'result.html', {'res':res})    

    return render(request,'home.html')   

def about(request):
    return render(request,'about.html')    