import os
from tqdm import tqdm

def find_sub_dirs(dir):
    sub_dirs = []
    for i in sorted(os.listdir(dir)):
        path = os.path.join(dir, i)
        if os.path.isdir(path):
            sub_dirs.append(path)
    return sub_dirs

if __name__ == '__main__':
    data_dir = '/mnt/ssd1/biomarker/c-met/data/LOGONE_AT2/patch'
    slide_pathes = find_sub_dirs(data_dir)

    for slide_path in slide_pathes:
        patch_pathes = find_sub_dirs(slide_path)
        print(slide_path)
        print(patch_pathes)
        for patch_path in tqdm(patch_pathes, desc = os.path.basename(slide_path), total = len(patch_pathes)):
            img_files = sorted([f for f in os.listdir(patch_path)])
            for img_file in img_files:
                if 'input' in img_file and 'png' in img_file:
                    os.remove(os.path.join(patch_path, img_file))
                    print(os.path.join(patch_path, img_file), '- Removed')
                elif 'label' in img_file and 'jpg' in img_file:
                    os.remove(os.path.join(patch_path, img_file))
                    print(os.path.join(patch_path, img_file), '- Removed')
                else:
                    continue



