# MLZoomcamp-2023 Week 1
This section includes the notes and development of the first week

## Notes

### 1. Setting the Virtual Machine with Amazon EC2
I created an Ubuntu EC2 instance and downoaded the key (.pem file) in the ~/.ssh folder

Then, tried to connect to the instance through the terminal with the following command:

```
ssh -i ~/.ssh/key_name.pem ubuntu@ipaddress 
```

The ip address comes from the EC2 instance at the AWS user interface

However I got the following error:

```
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
```

This error message indicates that the permissions of your private key file are too open. The recommended permissions for private key files are 400, meaning that only the owner has read and write permissions and no one else can access the file.

To fix this issue, you can change the permissions of the key file using the following command:

```
chmod 400 ~/.ssh/key_name.pem
```

Then run the instance connection again:

```
ssh -i ~/.ssh/key_name.pem ubuntu@ipaddress
```

I got good results. Now I can access with the ssh mlops-zoomcamp command

The course suggests to use Anaconda. I am not a personal fan but given that is an instance, I got it and gave it a try.

### 2. Installing Docker

```
sudo apt install docker.io
```

This went smoothly

### 3. Installing Docker compose

This was new to me so this is what the https://docs.docker.com/compose/ said about it

Compose is a tool for defining and running multi-container Docker applications. With Compose, you use a YAML file to configure your application’s services. Then, with a single command, you create and start all the services from your configuration.

Compose works in all environments: production, staging, development, testing, as well as CI workflows. It also has commands for managing the whole lifecycle of your application:

Start, stop, and rebuild services
View the status of running services
Stream the log output of running services
Run a one-off command on a service
The key features of Compose that make it effective are:

Have multiple isolated environments on a single host
Preserves volume data when containers are created
Only recreate containers that have changed
Supports variables and moving a composition between environments

Installing docker compose from github releases

```
wget https://github.com/docker/compose/releases/download/v2.18.0/docker-compose-linux-x86_64 -O docker-compose
```

```
chmod +x docker-compose to make it executable
```

Then I went to the main directory and added the path on the .bashrc

```
nano .bashrc
```

The going to the end and adding:

```
export PATH="{HOME}/soft:${PATH}"
```

soft is the directory where the docker-compose is stored

use docker without sudo

https://docs.docker.com/engine/install/linux-postinstall/

```
sudo groupadd docker
```

```
sudo usermod -aG docker $USER
```

logout and login again your virtual machine 

### 4. clone the repository and get access in VSCode
```
clone 'repo_url'
```
### 5. connect VScode to a virtual machine

Install the extension Remote - SSH

Open a Remote window

Connect to host. the configured ssh host mlops-zoomcamps should appear there or you can add a new one

You will get the full directory on VS code. Open the terminal, go to ports and add 8888

Now you can go to notebooks and type jupyter notebooks. The output will be a link that you type on your browser and will take you to the jupyter interface.
