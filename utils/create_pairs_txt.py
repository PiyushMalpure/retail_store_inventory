import os

def create_pairs_txt(query_folder_path, reference_folder_path, pairs_txt_path):
    """
    Create a pairs.txt file for SuperGlue
    Args:
        query_folder_path: str - path to the query folder
        reference_folder_path: str - path to the reference folder
        pairs_txt_path: str - path to save the pairs.txt file
    """
    query_images = os.listdir(query_folder_path)
    reference_images = os.listdir(reference_folder_path)
    
    with open(pairs_txt_path, 'w') as f:
        for query_image in query_images:
            for reference_image in reference_images:
                f.write(f"{query_image} {reference_image}\n")

if __name__ == '__main__':
    query_folder_path = '../results/predict/crops/retail' # Path to the cropped images
    reference_folder_path = '../data/images_inventory' # Path to the reference images
    pairs_txt_path = '../data/pairs.txt' # Path to save the pairs.txt file
    create_pairs_txt(query_folder_path, reference_folder_path, pairs_txt_path)
    