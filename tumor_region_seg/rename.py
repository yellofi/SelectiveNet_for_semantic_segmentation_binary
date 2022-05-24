import os

data_dir = '/mnt/hdd1/model/Lung_c-MET IHC_scored/UNet/06_baseline_samsung_data/1-fold/checkpoint'
idx = [i for i in range(10, 62)]

if __name__ == "__main__":

    for i in idx:

        old_name = os.path.join(data_dir, f'model_epoch20{i}.pth')
        new_name = os.path.join(data_dir, f'model_epoch2{i}.pth')

        print(old_name, new_name)

        os.rename(old_name, new_name)
