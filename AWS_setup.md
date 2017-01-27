sudo apt install python

curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"

sudo python get-pip.py

sudo pip install awscli

aws configure

Give the correct values:
Access key ID: ______
secret access key: _____
region name: us-east-1
output format: text

aws s3 sync s3://bucket local_folder

sudo pip install scipy
sudo pip install numpy
sudo pip install pandas
sudo pip install scikit-learn
