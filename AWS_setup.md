The basic unit I select is an Ubuntu machine mp.xlarge, and ran the following commands:
```
sudo apt install python

curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"

sudo python get-pip.py

sudo pip install awscli

aws configure

```

Then enter the correct values for:

Access key ID: ______
secret access key: _____
region name: us-east-1
output format: text

If we can see the buckets, we are good setting up:
```
aws s3 ls
```

To copy full folders from the buckets
```
aws s3 sync s3://bucket_name target_folder
```
Install specific packages

```
sudo pip install scipy

sudo pip install numpy

sudo pip install pandas

sudo pip install scikit-learn
```
