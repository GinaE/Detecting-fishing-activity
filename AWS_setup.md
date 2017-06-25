The unit I selected was an Ubuntu machine mp.xlarge, and ran the following commands:
```
sudo apt install python

curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"

sudo python get-pip.py

sudo pip install awscli

aws configure

```

After doing that you will be promted to enter the following info:

Access key ID: ______
secret access key: _____
region name: us-east-1
output format: text

If we can see the buckets, we are good with the set up:
```
aws s3 ls
```

To copy full folders from the buckets
```
aws s3 sync s3://bucket_name target_folder
```
To install specific packages, use the regular pip.

```
sudo pip install scipy

sudo pip install numpy

sudo pip install pandas

sudo pip install scikit-learn
```
