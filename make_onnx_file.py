import torch
import torch.onnx
from torch import nn
from torch.autograd import Variable
from UNet_3Plus import UNet_3Plus

def make_onnx_file(model, ckpt_path, save_path):
    dict_model = torch.load(ckpt_path)
    
    net = model
    net = nn.DataParallel(net).cuda()
    net.load_state_dict(dict_model['net'], strict=False)
    
    net.eval()
    
    # 모델 입력값
    batch_size, width, height = 1, 256, 256
    x = Variable(torch.randn(1, batch_size, height, width)).cuda()
    
    # 모델 변환
    torch.onnx.export(net.module,  # 실행될 모델
                  x,    # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  save_path,  # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,   # 모델 파일 안에 학습된 모델 가중치를 저장할지 여부
                  opset_version=14,     # 모델 변환 시 사용할 onnx 버전
                  do_constant_folding=True, # 최적화시 상수폴딩 사용여부
                  input_names=['input'],    # 모델 입력값명
                  output_names=['output'],  # 모델 출력값명
                  dynamic_axes={'input': {0:'batch_size', 2:'height', 3:'width'},   # 가변적인 길이를 가진 차원
                                'output':{0:'batch_size', 2:'height', 3:'width'}})
    


if __name__ == "__main__":
    model = UNet_3Plus(in_channels=1)
    ckpt_path = ''
    save_path = ''
    make_onnx_file(model, ckpt_path, save_path)