import requests
import os
import zipfile


def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def extract(file_path):
    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(os.path.dirname(file_path))
    zip_ref.close()


if __name__ == "__main__":
    glove_file_id = '1aKgkJWuU6gMyN_ZadJpJ6xdFaTiWrBsB'
    glove_destination = 'Datasets/Glove/glove.6B/glove.6B.50d.txt'
    model_small_id = '1OoifCcB7UQcHiavcRRMwP9wlqJK0hhek'
    model_small_destination = 'Models/model_large.hdf5'
    model_large_id = '1TCSdKA0l0vWz2b80sH1es7SsKUyp09MW'
    model_large_destination = 'Models/model_small.hdf5'
    dataset_train_id = '1B7QfJQ2JGWyzcfu51qzsdWGvMpe4_OTY'
    dataset_train_destination = 'Datasets/train-v2.0.json'
    dataset_dev_id = '1jI280ju5NE7CF2k7POPPuuNjg8mz4m-O'
    dataset_dev_destination = 'Datasets/dev-v2.0.json'

    if not os.path.exists('./Datasets/train-v2.0.json'):
        if not os.path.exists('./Datasets'):
            os.mkdir('./Datasets')
        print('Downloading train dataset..')
        download_file_from_google_drive(dataset_train_id, dataset_train_destination)
        print('Done.')

    if not os.path.exists('./Datasets/dev-v2.0.json'):
        if not os.path.exists('./Datasets'):
            os.mkdir('./Datasets')
        print('Downloading development dataset..')
        download_file_from_google_drive(dataset_dev_id, dataset_dev_destination)
        print('Done.')

    if not os.path.exists('./Datasets/Glove/glove.6B/glove.6B.50d.txt'):
        os.mkdir('./Datasets/Glove/glove.6B')
        print('Downloading glove 50d...')
        download_file_from_google_drive(glove_file_id, glove_destination)
        print('Done.')

    if not os.path.exists('./Models/model_large.hdf5'):
        os.mkdir('./Models/model_large.hdf5')
        print('Downloading pretrained-model(large)...')
        download_file_from_google_drive(model_large_id, model_large_destination)
        print('Done.')

    if not os.path.exists('./Models/model_small.hdf5'):
        os.mkdir('./Models/model_small.hdf5')
        print('Downloading pretrained-model(small)...')
        download_file_from_google_drive(model_small_id, model_small_destination)
        print('Done.')
