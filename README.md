# End-To-End-Wine-Quality-Machine-Learning-Project
This is End to End machine learning Project i need to use diffrent tech stacks including EC2,ECR,Github actions and S3-bucket , flask m html & css for ui


### Tool you have to install:-

1. Anaconda: https://www.anaconda.com/
2. Vs code: https://code.visualstudio.com/download
3. Git: https://git-scm.com/


### Data link:

- Kaggle: https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data




## How to run?

```bash
git clone https://github.com/IbraahimLab/End-To-End-Wine-Quality-Machine-Learning-Project
```

```bash
conda create -n wine python=3.10 -y
```

```bash
conda activate wine
```

```bash
pip install -r requirements.txt
```




# Create Mongo db ATLAS 

```bash
here : https://www.mongodb.com/products/platform/atlas-database

```

## 1. Login to AWS console.

# 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws

    3. S3 Fullacces


### Export the  environment variable
```bash


export MONGODB_URL="Your mongo db url"

export AWS_ACCESS_KEY_ID=<Your AWS_ACCESS_KEY_ID>

export AWS_SECRET_ACCESS_KEY=<Your AWS_SECRET_ACCESS_KEY>

```

## 3. Create ECR repo to store/save docker image
    - Save the URI: in temporary place you will put on github action 


## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

   - AWS_ACCESS_KEY_ID
   - AWS_SECRET_ACCESS_KEY
   - AWS_DEFAULT_REGION
   - ECR_REPO
   - MONGODB_URL
