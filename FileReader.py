import csv, sys
import numpy as np 

class FileReader(object): 
	'''
	class for parsing csv data files into lists and arrays of various dimensions
	- fileURL = file URL string 
	'''

	def __init__(self, fileURL): 
		self.fileURL = fileURL

	# function for parsing a csv file with 2D of ints into 2D list 
	def csv_to_list_ints(self): 
		listData = []
		try: 
			with open(self.fileURL, 'rU') as mf: 
				csvReader = csv.reader(mf)
				for row in csvReader: 
					listData.append([int(element) for element in row])
		except IOError: 
			print "file " + self.fileURL + " could not be found "
			sys.exit(1)
		return listData

	# function for parsing a csv file with 2D of ints into np array
	def csv_to_npArray_ints(self, np_dtype='int32'): 
		listData = self.csv_to_list_ints()
		return np.asarray(listData, dtype=np_dtype)

	# function for parsing a csv file with 1 row of strings into a 1D list
	def csv_1RowStr_to_list(self): 
		listData = []
		try: 
			with open(self.fileURL, 'rU') as mf: 
				csvReader = csv.reader(mf)
				for row in csvReader: 
					listData = row
		except IOError: 
			print "file " + self.fileURL + " could not be found "
			sys.exit(1)
		return listData
