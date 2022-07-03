from image_loader import *
from torch.autograd import Variable
import tqdm
import cv2
from model import *
from torchvision import transforms
import numpy as np
from misc import check_mkdir, crf_refine

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Interection=True
pattern= 'CN'
num_of_interaction = 4
fix_backbone = False
to_pil = transforms.ToPILImage()

input_size = 416

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

save_results_path = '/ghome/zhuyr/Results/ISTD/'
check_mkdir(save_results_path)
def test():
    net = SHADOW(backbone='efficientnet-b3',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=fix_backbone,
                 has_se=False,
                 num_of_layers=num_of_interaction,
                 pattern=pattern).to(device)
    net.load_state_dict(torch.load('/ghome/zhuyr/ACM MM22  Shadow Detection/CKPT_Shadow_Detection/ISTD.PTH'))
    net.eval()
    images_list = glob.glob('/ghome/zhuyr/ACM MM22  Shadow Detection/ISTD/test_A/*.*')
    for i, path in tqdm.tqdm(enumerate(images_list)):
        print('i-----------------:',i)
        img = Image.open(path)
        input_img = img
        W=img.size[0]
        H=img.size[1]
        img = Variable(transform(img).unsqueeze(0)).to(device)#.cuda()
        img,out_ns= net(img)
        img = torch.sigmoid(img)

        prediction = np.array(transforms.Resize((H, W))(to_pil(img.data.squeeze(0).cpu())))
        prediction = crf_refine(np.array(input_img.convert('RGB')), prediction)
        Image.fromarray(prediction).save(
            os.path.join(save_results_path, os.path.basename(path)[:-4:]) + '.png')

if __name__ == '__main__':
    test()
    res_root= save_results_path
    from Evaluation_codes import *
    gt_root = '/ghome/zhuyr/ACM MM22  Shadow Detection/ISTD/test_B'
    pos_err, neg_err, ber, acc, df = evaluate_pairs(res_root,gt_root)
    print(f'\t BER: {ber:.2f}, pErr: {pos_err:.2f}, nErr: {neg_err:.2f}, acc:{acc:.4f}')

