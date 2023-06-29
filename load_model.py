
import torch

from model import Deblur
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
	image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
	image = image.squeeze(0)  # remove the fake batch dimension
	image = unloader(image)
	plt.imshow(image)
	if title is not None:
		plt.title(title)
	plt.pause(5)  # pause a bit so that plots are updated

def loadModel(inputImage):
	model = Deblur()
	model.load_state_dict(torch.load('.\\latest.pth', map_location=torch.device('cpu')))
	img = Image.open(inputImage).convert('RGB')
	print(img.size)
	img_size = img.size
	swapped_size = (img_size[1], img_size[0])
	trans = transforms.Compose([
   		transforms.CenterCrop(swapped_size),
    	#transforms.Resize(256),
    	#transforms.GaussianBlur(21),
    	transforms.ToTensor()
	])
	model.eval()
	img = trans(img)
	#imshow(img)
	return model(img)
	#imshow(out)

#loadModel(r"C:\Users\jafle\AI-Deblur\data\0_IPHONE-SE_S.JPG")
