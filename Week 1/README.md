# MLZoomcamp-2023 Week 1
This section includes the notes and development of the first week

## Notes

### 1. Setting the Virtual Machine with Amazon EC2
I created an Ubuntu EC2 instance and downoaded the key (.pem file) in the ~/.ssh folder

Then, tried to connect to the instance through the terminal with the following command:
ssh -i ~/.ssh/key_name.pem ubuntu@ipaddress 

The ip address comes from the EC2 instance at the AWS user interface

However I got the following error:

The authenticity of host 'X.xxx.xxx.xxx (X.xxx.xxx.xxx)' can't be established.
ECDSA key fingerprint is SHA256:0WIGHcYT5i+lBvTKWwV360Jfjg9+VC8QoI6upTNJlhY.
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added 'X.xxx.xxx.xxx' (ECDSA) to the list of known hosts.
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@         WARNING: UNPROTECTED PRIVATE KEY FILE!          @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Permissions 0644 for '/Users/user_name/.ssh/key_name.pem' are too open.
It is required that your private key files are NOT accessible by others.
This private key will be ignored.
Load key "/Users/user_name/.ssh/key_name.pem": bad permissions
ubuntu@X.xxx.xxx.xxx: Permission denied (publickey).

This error message indicates that the permissions of your private key file are too open. The recommended permissions for private key files are 400, meaning that only the owner has read and write permissions and no one else can access the file.

To fix this issue, you can change the permissions of the key file using the following command:

chmod 400 ~/.ssh/key_name.pem

Then run the instance connection again:

ssh -i ~/.ssh/key_name.pem ubuntu@ipaddress

I got good results. Now I can access with the ssh mlops-zoomcamp command

The course suggests to use Anaconda. I am not a personal fan but given that is an instance, I got it and gave it a try.

## 2. Installing Docker

sudo apt install docker.io