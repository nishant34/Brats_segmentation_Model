import torch.utils.data as dataloader
from dataloader import H5Dataset
import torch.optim as optim
from common import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_S = FCN1(num_input_features=32,drop_rate=0.2,num_classes=4).to(device)


criterion_S = nn.CrossEntropyLoss().cuda()


optimizer_S = optim.Adam(model_S.parameters(), lr=lr_S, weight_decay=6e-4, betas=(0.97, 0.999))
scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=step_size_S, gamma=0.1)



if __name__ == '__main__':
    #train data
    mri_data_train = H5Dataset("/train_data", mode='train')
    trainloader = dataloader.DataLoader(mri_data_train, batch_size=8, shuffle=True)
    #val data
    mri_data_val = H5Dataset("/val_data", mode='val')
    valloader = dataloader.DataLoader(mri_data_val, batch_size=2, shuffle=False)
    
    print('Rate     | epoch  | Loss seg| DSC_val')
    for epoch in range (num_epoch):
        scheduler_S.step(epoch)
        
        model_S.train()
        for i, data in enumerate(trainloader):
            images, targets = data
            
            images = images.to(device)
            targets = targets.to(device)
                   
            optimizer_S.zero_grad()
            outputs = model_S(images)
            loss_seg = criterion_S(outputs, targets) 
            loss_seg.backward()
            optimizer_S.step()

        
        with torch.no_grad():
            for data_val in valloader:
                images_val, targets_val = data_val
                model_S.eval()
                images_val = images_val.to(device)
                targets_val = targets_val.to(device)

                outputs_val = model_S(images_val)
                _, predicted = torch.max(outputs_val.data, 1)
                
                predicted_val = predicted.data.cpu().numpy()
                targets_val = targets_val.data.cpu().numpy()
                dsc = []
                for i in range(1, num_classes): 
                    dsc_i = dice(predicted_val, targets_val, i)
                    dsc.append(dsc_i)
                dsc = np.mean(dsc)

                

        
        for param_group in optimizer_S.param_groups:
            print('%0.6f | %6d | %0.5f | %0.5f ' % (\
                    param_group['lr'], epoch,
                   
                    loss_seg.data.cpu().numpy(),
                    
                    dsc))

        
        if (epoch % step_size_S) == 0 or epoch == (num_epoch - 1) or (epoch % 100) == 0:
            torch.save(model_S.state_dict(), '/model/savedstate' + '%s_%s.pth' % (str(epoch).zfill(5), checkpoint_name))

