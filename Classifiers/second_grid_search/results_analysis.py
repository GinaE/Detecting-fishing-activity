class model_results(object):
	"""docstring for model_results"""
	def __init__(self,filename):
		# super(model_results, self).__init__()
		self.name = None
		self.best_params = None
		self.train_acc = None
		self.test_acc = None
		self.train_f1 = None
		self.test_f1 = None
		self.top_features = None

	def read_results(self,filename):
		#   datasets[name] = dict(zip(['all', 'train', 'cross', 'test'], load_dataset_by_vessel(os.path.join(dir, filename))))
		f=open('filename')
		lines=f.readlines()
		line7 = lines[7].strip().split()
		self.name = line7[3]
		self.best_params = lines[8].strip()

		line13 = lines[13].strip().split()
		line14 = lines[14].strip().split()
		self.train_acc = line13[2]
		self.test_acc = line13[6]
		self.train_f1 = line14[2]
		self.test_f1 = line14[6]
		self.top_features = [item.strip() for item in lines[18:]]

# datasets = {}
for filename in os.listdir('results'):
	if filename.endswith('.txt'):
		name = filename[:-len('.txt')]


