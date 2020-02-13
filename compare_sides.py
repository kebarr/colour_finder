import numpy as np
from scipy.ndimage.morphology import binary_dilation
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import measure
import os
from skimage import morphology




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
        self.conditions = conditions

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
                pass



class ComparisonResult:
    def __init__(self, name):
        self.name = name
        self.total_intensity_diffs = {}
        self.inner_intensity_diffs = {}
        self.prop_diffs = {}
        self.average_total_intensity_diff = -1
        self.average_inner_intensity_diff = -1
        self.average_prop = -1



class Stain:
    def __init__(self, side_folders, name, conditions):
        self.side_folders = side_folders
        self.name = name
        self.conditions = conditions
        print(side_folders)
        self.get_filenames_and_create_animals()

    def get_filenames_and_create_animals(self):
        animals = {condition:{} for condition in self.conditions}
        # assumes that file system is of the form:
        # folder/stain_name/condition
        for folder in self.side_folders:
            print(folder)
            for condition in self.conditions:
                folder_path = folder + '/' + self.name + '/' + condition
                files = os.listdir(folder_path)
                animal_names = set([s.split(' ')[0] for s in files if s.endswith('.tif')])
                for animal_name in animal_names:
                    print(animals[condition].keys())
                    if animal_name not in animals[condition].keys():
                        animals[condition][animal_name] = {}
                    animal = Animal(folder_path, animal_name)
                    animals[condition][animal_name][folder] = animal
            print(animals)
        self.animals = animals

    def run(self):
        for condition in self.conditions:
            print("running for condition %s" % (condition))
            for animal in self.animals[condition]:
                for animal_side in self.animals[condition][animal]:
                    print(animal_side)
                    self.animals[condition][animal][animal_side].analyse_images()




# try again but with stain helper class
class CompareSides:
    def __init__(self, folder_side_one, folder_side_two, list_of_stain_names, list_of_conditions):
        self.folder_side_one = folder_side_one
        self.folder_side_two = folder_side_two
        self.list_of_stain_names = list_of_stain_names
        self.list_of_conditions = list_of_conditions
        self.create_stains()

    def create_stains(self):
        stains = []
        for stain in self.list_of_stain_names:
            stains.append(Stain([self.folder_side_one, self.folder_side_two], stain, self.list_of_conditions))
        self.stains = stains

    def run(self):
        for stain in self.stains:
            print("running for stain %s" % (stain.name))
            stain.run()

    def perform_comparison(self):
        final = {}
        # compare the probe and contralateral sides of each sample
        for stain in self.stains:
            final[stain.name] = {}
            for condition in stain.conditions:
                final[stain.name][condition] = {}
                for animal in stain.animals[condition]:
                    print(animal)
                    # if both sides are present, so we can actually do comparison
                    if len(stain.animals[condition][animal].keys()) == 2:
                        side1 = self.folder_side_one
                        side2 = self.folder_side_two
                        print(stain.animals[condition][animal][side1].samples)
                        animal_side1 = stain.animals[condition][animal][side1]
                        animal_side2 = stain.animals[condition][animal][side2]
                        side1_samples = animal_side1.samples
                        side2_samples = animal_side2.samples
                        res = ComparisonResult(animal)
                        for i, sample in enumerate(side1_samples):
                            if sample in side2_samples:
                                print(sample)
                                j = side2_samples.index(sample)
                                result1 = stain.animals[condition][animal][side1].result
                                result2 = stain.animals[condition][animal][side2].result
                                total_intensity_diff = result1.total_intensities[i] - result2.total_intensities[j]
                                inner_intensity_diff = result1.inner_intensities[i] - result2.inner_intensities[j]
                                prop_diff = result1.props[i] - result2.props[j]
                                res.total_intensity_diffs[sample] = total_intensity_diff
                                res.inner_intensity_diffs[sample] = inner_intensity_diff
                                res.prop_diffs[sample] = prop_diff
                        total_averaged_diff = result1.total_averaged - result2.total_averaged
                        inner_averaged_diff = result1.inner_averaged - result2.inner_averaged
                        prop_averaged_diff = result1.prop_averaged - result2.prop_averaged
                        res.average_total_intensity_diff = total_averaged_diff
                        res.average_inner_intensity_diff = inner_averaged_diff
                        res.average_prop = prop_averaged_diff
                        final[stain.name][condition][animal] = res
        self.final = final

    def output_result(self, out_filename_base):
        for stain in self.list_of_stain_names:
            stain_res = self.final[stain]
            print("stain res")
            print(stain_res)
            # create array of values to output
            res_prop_diffs = [[] for i in range(len(stain_res.keys()))]
            res_inner_diffs = [[] for i in range(len(stain_res.keys()))]
            res_total_diffs = [[] for i in range(len(stain_res.keys()))]
            res_prop_average_diffs = [[] for i in range(len(stain_res.keys()))]
            res_inner_average_diffs = [[] for i in range(len(stain_res.keys()))]
            res_total_average_diffs = [[] for i in range(len(stain_res.keys()))]
            for i, condition in enumerate(stain_res.keys()):
                condition_res = stain_res[condition]
                for animal in condition_res.keys():
                    animal_res = condition_res[animal]
                    for sample in animal_res.prop_diffs.values():
                        res_prop_diffs[i].append(sample)
                    for sample in animal_res.inner_intensity_diffs.values():
                        res_inner_diffs[i].append(sample)
                    for sample in animal_res.total_intensity_diffs.values():
                        res_total_diffs[i].append(sample)
                    print(res_prop_diffs)
                    res_prop_average_diffs[i].append(animal_res.average_prop)  
                    res_inner_average_diffs[i].append(animal_res.average_inner_intensity_diff)  
                    res_total_average_diffs[i].append(animal_res.average_total_intensity_diff)                
            out_filename_prop_diffs = out_filename_base + '_' + stain + '_prop_diffs.csv'                
            out_filename_inner_diffs = out_filename_base + '_' + stain + '_inner_diffs.csv'                
            out_filename_total_diffs = out_filename_base + '_' + stain + '_total_diffs.csv'                
            out_filename_prop_diffs_average = out_filename_base + '_' + stain + '_average_prop_diffs.csv'                
            out_filename_inner_average_diffs = out_filename_base + '_' + stain + '_inner_average_prop_diffs.csv'                
            out_filename_total_average_diffs = out_filename_base + '_' + stain + '_total_average_prop_diffs.csv'                
            res_prop_diffs = list(map(list, zip(*res_prop_diffs)))
            res_inner_diffs = list(map(list, zip(*res_inner_diffs)))
            res_total_diffs = list(map(list, zip(*res_total_diffs)))
            print(res_prop_average_diffs)
            #res_prop_average_diffs = list(map(list, zip(*res_prop_average_diffs)))
            #res_inner_average_diffs = list(map(list, zip(*res_inner_average_diffs)))
            #res_total_average_diffs = list(map(list, zip(*res_total_average_diffs)))
            # need to output csv in format ready for estimation stats.
            np.savetxt(out_filename_prop_diffs, np.array([np.array(xi) for xi in res_prop_diffs]), header='2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene')
            np.savetxt(out_filename_inner_diffs, np.array([np.array(xi) for xi in res_inner_diffs]).transpose(), header='2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene')
            np.savetxt(out_filename_total_diffs, np.array([np.array(xi) for xi in res_total_diffs]).transpose(), header='2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene')
            np.savetxt(out_filename_prop_diffs_average, np.array([np.array(xi) for xi in res_prop_average_diffs]).transpose(), header='2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene')
            np.savetxt(out_filename_inner_average_diffs, np.array([np.array(xi) for xi in res_inner_average_diffs]).transpose(), header='2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene')
            np.savetxt(out_filename_total_average_diffs, np.array([np.array(xi) for xi in res_total_average_diffs]).transpose(), header='2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene')


compare_sides = CompareSides('probe_side', 'contralateral', ['GFAP', 'IBA1'], ['2weekDummy', '2weekGraphene', '6weekDummy'])
compare_sides.run()
compare_sides.perform_comparison()
compare_sides.output_result('test')



