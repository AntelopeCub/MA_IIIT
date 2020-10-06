from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def file_upload(upload_file_list, folder_id='1HE8bkkl9MVvPCYAGpoA9Y6pKhjeVVUOH'):
    # Rename the downloaded JSON file to client_secrets.json
    # The client_secrets.json file needs to be in the same directory as the script.
    gauth = GoogleAuth()
    drive = GoogleDrive(gauth)

    # Upload files to your Google Drive
    for upload_file in upload_file_list:
        gfile = drive.CreateFile({'parents': [{'id': folder_id}]})
        # Read file and set it as a content of this instance.
        gfile.SetContentFile(upload_file)
        gfile.Upload() # Upload the file.


if __name__ == "__main__":
    file_upload(['./models/vgg16/vgg16_bn_128_norm_SGDNesterov_l2=0.0005_avg_cifar_auto_237_0.9561_weights.h5'])