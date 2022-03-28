# Data analysis
- Document here the project: ser
- Description: Project Description
- Data Source:
- Type of analysis:

Please document the project the better you can.
# Enviorment Variables

You need to set up enviroment variables for the makefile and docker deployment to work.
- In you terminal enviroment
  - PROJECT_ID - Name of your GCP Project ID
  - DOCKER_IMAGE_NAME - Name you want to give the image which will found in GCP contain service.
  - GOOGLE_APPLICATION_CREDENTIALS - You have to make sure you download the json file and and set this variable to where you have it on your enviroment. ALWAYS store in a place which your environment can only access. (e.g ~/.gcpcred)
- In the Makefile
  - REGION - GCP region
  - BUCKET_NAME - Name of bucket to store training data and various files


```bash
export PROJECT_ID=<name of GCP Project Id>
export DOCKER_IMAGE_NAME=<Name>
export GOOGLE_APPLICATION_CREDENTIALS=<gcp creditonals file>
```
# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for ser in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/ser`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "ser"
git remote add origin git@github.com:{group}/ser.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
ser-run
```

# Install

Go to `https://github.com/{group}/ser` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/ser.git
cd ser
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
ser-run
```
# Lastly
This repo you can use to delploy a docker image and run on GCP only.  There is a addtional git repo which has a front end (Streamlit) https://github.com/Iases22/ser_app if you download and change a "url" varible in the app.py file.
