import urllib.request

print('Beginning file download with urllib2...')

url = 'https://hcp-wumann-data.s3.eu-west-2.amazonaws.com/rest-phaseOne.nii.gz'
urllib.request.urlretrieve(url, 'rest-phaseOne.nii.gz')

url = 'https://hcp-wumann-data.s3.eu-west-2.amazonaws.com/task.nii.gz'
urllib.request.urlretrieve(url, 'task.nii.gz')
