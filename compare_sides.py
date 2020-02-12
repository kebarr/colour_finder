import numpy as np
from scipy.ndimage.morphology import binary_dilation
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import measure
import os
from skimage import morphology
import os # imports functions that enable you to access the operating system, for example listing the contents of directories
import javabridge # enables you to run java code
import bioformats as bf # needed to read microscopy image files
from xml.etree import ElementTree as ETree # read xml structure, in our case, the .lif files
javabridge.start_vm(class_path=bf.JARS) 
import sklearn.preprocessing as preprocessing

# store all numbers for each animal together
class Result:
    def __init__(self):
        self.total_intensities = []
        self.inner_intensities = []
        self.props = []

    def average(self):
        self.total_averaged = np.mean(self.total_intensities)
        self.inner_averaged = np.mean(self.inner_intensities)
        self.prop_averaged = np.mean(self.props)

class Animal:
    def __init__(self, containing_folder, name):
        self.containing_folder = containing_folder
        self.name = name
        self.samples = [sample for sample in os.listdir(containing_folder) if name in sample]
        self.result = Result()

    def analyse_image(self, filename):
        file_path = self.containing_folder + '/' + filename
        im = Image.open(file_path).convert('L')
        data = np.array(im, dtype=float)
        norm = preprocessing.normalize(im, norm='l2')
        inner_intensity = np.sum(norm[246:-246, 420:-420])
        total_intensity = np.sum(norm)
        prop = float(inner_intensity)/float(total_intensity)
        self.result.inner_intensities.append(inner_intensity)
        self.result.total_intensities.append(total_intensity)
        self.result.props.append(prop)
        print('results for %s: inner intensity %d, total intensity %d, prop %f' % (filename, inner_intensity, total_intensity, prop))

    def analyse_images(self):
        print('analysing animal %s' % (self.name))
        for sample in self.samples:
            self.analyse_image(sample)
        self.result.average()

# helper class to store data from each side
# want to store results for each, should include animal and sample number too
class Side:
    def __init__(self, folder, stains, conditions):
        self.folder = folder
        self.stains = stains
        self.conditions = conditions
        self.get_filenames_and_create_animals()

    def get_filenames_and_create_animals(self):
        animals = {stain:{condition:[] for condition in self.conditions} for stain in self.stains} 
        # assumes that file system is of the form:
        # folder/stain_name/condition
        number_of_animals = 0
        for stain in self.stains:
            for condition in self.conditions:
                folder_path = self.folder + '/' + stain + '/' + condition
                files = os.listdir(folder_path)
                animal_names = set([s.split(' ')[0] for s in files if s.endswith('.tif')])
                number_of_animals += len(animal_names)
                animals_in_this_condition = []
                for animal_name in animal_names:
                    animal = Animal(folder_path, animal_name)
                    animals_in_this_condition.append(animal)
                animals[stain][condition] = animals_in_this_condition
        self.animals = animals

    def run(self):
        for stain in self.stains:
            for condition in self.conditions:
                for animal in self.animals[stain][condition]:
                    animal.analyse_images()




# need to compare contralateral side to probe site side to quantify differences

class CompareSides:
    def __init__(self, folder_side_one, folder_side_two, list_of_stain_names, list_of_conditions):
        self.folder_side_one = folder_side_one
        self.folder_side_two = folder_side_two
        self.list_of_stain_names = list_of_stain_names
        self.list_of_conditions = list_of_conditions
        self.create_sides()

    def create_sides(self):
        self.side_one = Side(self.folder_side_one, self.list_of_stain_names, self.list_of_conditions)
        self.side_two = Side(self.folder_side_one, self.list_of_stain_names, self.list_of_conditions)

    def run(self):
        print("running side one %s" % (self.folder_side_one))
        self.side_one.run()
        print("running side two %s" % (self.folder_side_two))
        self.side_two.run()

    def perform_comparison(self):
        # compare the probe and contralateral sides of each sample
        for stain in self.list_of_stain_names:
            for condition in self.list_of_conditions:
                # need to match each animal name and then each section



two_wk_dummy = 'slidescanner_images/contralateral/2 week Dummy/'
two_wk_dummy_files = os.listdir('slidescanner_images/contralateral/2 week Dummy')

for f in two_wk_dummy_files:
    file_path = two_wk_dummy + f
    if file_path.endswith('GFAP.tif'):
        im = Image.open(file_path).convert('L')
        data = np.array(im, dtype=float)
        print(f)
        norm = preprocessing.normalize(im, norm='l2')
        intensity_middle = np.sum(norm[246:-246, 420:-420])
        total_intensity = np.sum(norm)
        prop = float(intensity_middle)/float(total_intensity)
        print("middle intensity %d total intensity %d prop %f norm %d " % (intensity_middle, total_intensity, prop, np.linalg.norm(norm)))
