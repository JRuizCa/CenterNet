import json
import numpy as np
import pandas as pd


def transform_preds_results_json_to_txt(results_json):
    with open(results_json) as json_file:
        data = json.load(json_file)
    data = pd.DataFrame(data)
    data['bbox'] = data['bbox'].apply(lambda x: [x[0], x[1], np.round(x[0]+x[2], 3), np.round(x[1]+x[3], 3)])
    data['category_id'] = data['category_id'].apply(lambda x: 'barcode')
    data.set_index('image_id', inplace=True)
    for index in data.index.unique():
        with open(f'../mAP/input/detection-results/{index}.txt', 'w') as txt_file:
            for i in range(len(data[data.index == index])):
                data_index = data[data.index == index]
                txt_file.write(f"barcode ")
                txt_file.write(f"{data_index['score'].iloc[i]} ")
                txt_file.write(f"{data_index['bbox'].iloc[i][0]} {data_index['bbox'].iloc[i][1]} {data_index['bbox'].iloc[i][2]} {data_index['bbox'].iloc[i][3]} ")
                txt_file.write("\n")
    txt_file.close()

def transform_gt_json_to_txt(gt_json):
    with open(gt_json) as json_file:
        data = json.load(json_file)
    data = pd.DataFrame(data['annotations'])
    data['bbox'] = data['bbox'].apply(lambda x: [x[0], x[1], x[0]+x[2], x[1]+x[3]])
    data['category_id'] = data['category_id'].apply(lambda x: 'barcode')
    data.set_index('image_id', inplace=True)
    for index in data.index.unique():
        with open(f'../mAP/input/ground-truth/{index}.txt', 'w') as txt_file:
            for i in range(len(data[data.index == index])):
                data_index = data[data.index == index]
                txt_file.write(f"barcode ")
                txt_file.write(f"{data_index['bbox'].iloc[i][0]} {data_index['bbox'].iloc[i][1]} {data_index['bbox'].iloc[i][2]} {data_index['bbox'].iloc[i][3]} ")
                txt_file.write("\n")
        txt_file.close()


def main():
    transform_preds_results_json_to_txt('/home/julia/TFM/CenterNet/exp/ctdet/barcode_hg/results.json')
    transform_gt_json_to_txt('/home/julia/TFM/CenterNet/data/barcode/annotations/instances_test.json')
    print("Done!")
        
if __name__ == "__main__":
    main()