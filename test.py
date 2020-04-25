import torch
import os
from models import UNet
from data_util import MyDataset_test, MyDataset_test_moving
from functions import convert_im


def test(opt, log_dir, generator = None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if generator == None:
        generator = UNet(opt.sample_num, opt.channels, opt.batch_size, opt.alpha)
        
        checkpoint = torch.load(opt.load_model, map_location=device)
        generator.load_state_dict(checkpoint['g_state_dict'])
        del checkpoint
        torch.cuda.empty_cache()
    
    generator.to(device)
    generator.eval()

    dataloader = torch.utils.data.DataLoader(MyDataset_test(opt), opt.batch_size, shuffle=True, num_workers=0)

    for i, (imgs, filename) in enumerate(dataloader):
            with torch.no_grad():
                test_img = generator(imgs.to(device))
                filename = filename[0].split('/')[-1]
                filename = "test/" + filename +'.png'
                test_img = convert_im(test_img, os.path.join(log_dir ,filename), nrow=5, normalize=True, save_im=True)

def test_moving(opt, log_dir, generator = None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    if generator == None:
        generator = UNet(opt.sample_num, opt.channels, opt.batch_size, opt.alpha)
        
        checkpoint = torch.load(opt.load_model, map_location=device)
        generator.load_state_dict(checkpoint['g_state_dict'])
        del checkpoint
        torch.cuda.empty_cache()
    
    generator.to(device)
    generator.eval()

    dataloader = torch.utils.data.DataLoader(MyDataset_test_moving(opt), opt.batch_size, shuffle=True, num_workers=opt.num_workers_dataloader)

    for i, (imgs, filename) in enumerate(dataloader):
            with torch.no_grad():
                filename = filename[0].split('/')[-1]
                for k in range(len(imgs)):
                    test_img = generator(imgs[k].to(device))
                    folder_path = os.path.join(log_dir, "test/%s" % filename)
                    os.makedirs(folder_path, exist_ok=True)
                    filename_ = filename + '_' + str(k) +'.png'
                    test_img = convert_im(test_img, os.path.join(folder_path ,filename_), nrow=5, normalize=True, save_im=True)
                    





