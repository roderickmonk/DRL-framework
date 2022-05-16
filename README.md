# CO3 Module

'CO3' is the project mneumonic that is used throughout.

![Image](./Images/CO3_Repos.png "A View of All CO3-related Repos")

## Installation

### Preparation

Ensure that the following environment variables are defined. For suggestions as to where to set these environment variables, refer to the following document: https://help.ubuntu.com/community/EnvironmentVariables.

    MONGODB   # This is the connect string to the database
    CO3_PATH  # This variable defines the location where the co3 repo is cloned

Note: The MONGODB connect string has the form: mongodb://<username>:<password>@<database_host>:<port>/admin?readPreference=primary

### Python Version

The specific Python version may advance throughout the life of the project and will be noted here. Currently the recommended version is `3.9.10`. Only the first 2 numbers are needed to refer to the Python run-time; in order to avoid duplication, a generic Python version will be referred to below as pythonX.Y.

It has been found that the safest and most reliable means to install/upgrade Python is from source. The reader is invited to google "ubuntu install pythonX.Y" for the specifics as to how to install from source.

## Software Installation

It is assumed that required Python version has already been installed on the target ec2. What follows are instructions to install the co3 module.

1.  Clone the co3 repo:

        $ cd ~
        $ git clone git@github.com:1057405bcltd/co3.git

    It is assumed in the above that the target folder is `~/co3` and that the environment variable CO3_PATH has been configured accordingly. If the repo has been cloned into some other folder, then CO3_PATH must be set accordingly.

2.  Create a Python virtual environment in the same folder:

        $ cd $CO3_PATH
        $ pythonX.Y -m venv .

3.  and then activate the new virtual environment:

        $ source project

4.  Install the required Python modules:

        $ pip install -r requirements.txt --use-feature=2020-resolver

5.  Install redis on Ubuntu 20.04: https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-20-04

    Install redis on Ubuntu 18.04: https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-18-04

6.  Install gym environments:

        $ cd $CO3_PATH
        $ pip install -e co3/environments

7.  Install mongodb locally:

        # Set LOCALDB in /etc/environment

        cd /shared
        mkdir mongodb_data
        docker run -p 27017:27017 -v /shared/mongodb_data:/data/db \
        -e MONGO_INITDB_ROOT_USERNAME=<usermame> \
        -e MONGO_INITDB_ROOT_PASSWORD=<password> \
        --restart unless-stopped \
        --name local_mongodb -d mongo:4.4.3-bionic 

8.  Things to do to access a server from github actions:
    
    * Execute the following commands:
  
        $ cd ~/.ssh; ssh-copy-id -f -i github_actions <server_name>

    * Add the IP address from ~/.ssh/config and use it to create a new github secret <SERVER_NAME> (all upper case)

    * Thereafter refer to the server in GitHub actions as <SERVER_NAME>


9.  Explicitly set an AWS ec2 as headless:

    sudo apt-get install xserver-xorg libglu1-mesa-dev freeglut3-dev mesa-common-dev libxmu-dev libxi-dev
    sudo nvidia-xconfig -a --use-display-device=None --virtual=1280x1024
    sudo /usr/bin/X :0 &

    Then add the following to /etc/environment
    DISPLAY=:0


## Software Update

1.  Pull from git:

        $ cd $CO3_PATH
        $ source project
        $ git pull

2.  Ensure that python modules are up to date:

        $ upgrade-modules

## RStudio

1.  How to Install Docker On Ubuntu 20.04

        https://linuxconfig.org/how-to-install-docker-on-ubuntu-20-04-lts-focal-fossa

2.  Starting RStudio (assuming Docker has been installed)

        $ cd /shared/co3
        $ docker run -d -p 8787:8787 -v $(pwd):/home/rstudio -e PASSWORD=sentient --restart unless-stopped rocker/rstudio

## Running

Execution is initiated as follows:

    $ cd $CO3_PATH
    $ source project
    $ co3

Note that a number of configuration parameters are available. A complete list can be found in `./docs/co3-configuration.pdf`.

## Outputs

1.  `logs`: For each run, a new log file is created in the folder CO3_PATH/logs. The names of log files have the form 'YYYY-MM-DDTHH:MM:SS.log'. Log file content mirrors the console log messages.

2.  `rewards`: For each run, a new CSV file is created containing a history of the rewards that were allocated to each episode and step within each episode. Each line of the file contains the episode number, the step number, and the reward. Note that the name of the file is of the form 'YYYY-MM-DDTHH:MM:SS.csv'. All such files are recorded into the folder \$CO3_PATH/rewards.

3.  `networks`: During execution and at the end of each episode the policy is retrained and a snapshot of the new policy is recorded to the file: `$CO3_PATH/policy/policy.pt` Note: each snapshot overwrites the previous.

4.  `datasets`: Because the determination of state is an expensive operation, it is possible to save each state to the database and then replay a run without the need to recalculate each state. Refer to the following command line parameters: `db-action` and `state-name`.

## Cleaning

Remove unused imports and variables

        $ autoflake -i -r --remove-all-unused-imports --remove-unused-variables co3

## Testing

It is always worthwhile when taking delivery of new software to be sceptical. A number of pytests have ben defined and these can be carried out as follows:

    $ cd $CO3_PATH
    $ source project
    $ pytest co3

Should a test fail, report the number of the failed test case to DEV.

