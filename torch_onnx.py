import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, utils, models
from efficientnet_pytorch import EfficientNet
import onnx
import onnxruntime


def softmax(x):
	x = x.reshape(-1)
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum(axis=0) 
    
    

class MODEL:
    def __init__(self):
        self.num_cls = 5
        imag_size = 224
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        self.model = EfficientNet.from_name('efficientnet-b0')
        self.model = self.change_output_layers(self.model).cuda()
        # print(self.model)
        
        
    def change_output_layers(self, model):
        model._fc = nn.Linear(1280, self.num_cls, bias=True)
        return model
        
              
    
def convert():
    dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
    net = MODEL()
    net.model.load_state_dict(torch.load('weights.pkl'))
    net.model.eval()
    net.model.set_swish(memory_efficient=False)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(net.model, dummy_input, "torch_efficientnet.onnx", verbose=True, input_names=input_names, output_names=output_names)


    
def inference(filename):
    
    sess = onnxruntime.InferenceSession("torch_efficientnet.onnx")
    sess.set_providers(['CPUExecutionProvider'])
    
    # input name and shape
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    # output name
    output_name = sess.get_outputs()[0].name  
    
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
    
    #----------------------load image----------------------#
    img = cv2.imread(filename,0)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype('uint8')
    img = transform(img)
    img = img.view(-1,3,224,224)
    img = img.numpy().astype(np.float32)
    #----------------------load image----------------------#
    
    
    outputs = sess.run([output_name], {input_name: img})[0]
    outputs = softmax(np.array(outputs)) # not necessary
    
    print("predicted result = ",outputs)
    print('finifhsed\n\n')
    


if __name__ == '__main__':
    convert() # step1: convert model to onnx
    inference('WL_s28.bmp') # step2: import onnx model and do prediction
    
    
    