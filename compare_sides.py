import numpy as np
from scipy.ndimage.morphology import binary_dilation
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import measure
import os
from skimage import morphology
from itertools import zip_longest



# store all numbers for each animal together
class Result:
    def __init__(self, metrics):
        self.metrics = metrics
        self.results = [[] for i in range(len(metrics))]
        self.results_averaged = [0 for i in range(len(metrics))]

    def average(self):
        for i in range(len(self.metrics)):
            self.results_averaged[i] = np.mean(self.results[i])


def inner_intensity(data):
    return np.sum(data[246:-246, 420:-420])


def prop(data):
    inner_intensity = np.sum(data[246:-246, 420:-420])
    total_intensity = np.sum(data)
    return float(inner_intensity)/float(total_intensity)

class Animal:
    def __init__(self, containing_folder, name, metrics):
        self.containing_folder = containing_folder
        self.name = name
        self.samples = [sample for sample in os.listdir(containing_folder) if name in sample]
        self.result = Result(metrics)

    def analyse_image(self, filename):
        file_path = self.containing_folder + '/' + filename
        im = Image.open(file_path).convert('L')
        data = np.array(im, dtype=float)
        norm = preprocessing.normalize(im, norm='l2')
        for i, metric in enumerate(self.result.metrics):
            res = metric.function(norm)
            self.result.results[i].append(res)
            print('results for %s metric %d: %d' % (filename, res))

    def analyse_images(self):
        print('analysing animal %s' % (self.name))
        for sample in self.samples:
            self.analyse_image(sample)
        self.result.average()



class Stain:
    def __init__(self, side_folders, name, conditions, list_of_metrics):
        self.side_folders = side_folders
        self.name = name
        self.conditions = conditions
        self.list_of_metrics = list_of_metrics
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
                    # shouldn't actually need to wire metrics through everything like this
                    animal = Animal(folder_path, animal_name, self.list_of_metrics)
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


# makes sense to just have one metric object, which is reused for each animal 
class Metric:
    def __init__(self, name, function):
        self.name = name
        self.function = function

    def write_output_string(self, header, data):
        ''' convert data to string to write to results csv '''
        res = list(map(list, zip_longest(*data)))
        out_str = 'prop diffs:\n'
        out_str += '2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene\n'
        for i in range(len(res)):
            for j in range(len(res[i])):
                out_str += str(res[i][j])+', '
            out_str += '\n'
        out_str +='\n'
        return out_str

# try again but with stain helper class
class CompareSides:
    def __init__(self, folder_side_one, folder_side_two, list_of_stain_names, list_of_conditions, list_of_metrics):
        self.folder_side_one = folder_side_one
        self.folder_side_two = folder_side_two
        self.list_of_stain_names = list_of_stain_names
        self.list_of_conditions = list_of_conditions
        self.list_of_metrics = list_of_metrics
        self.create_stains()

    def create_stains(self):
        stains = []
        for stain in self.list_of_stain_names:
            stains.append(Stain([self.folder_side_one, self.folder_side_two], stain, self.list_of_conditions, self.list_of_metrics))
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
                    res_prop_average_diffs[i].append(animal_res.average_prop)  
                    res_inner_average_diffs[i].append(animal_res.average_inner_intensity_diff)  
                    res_total_average_diffs[i].append(animal_res.average_total_intensity_diff)                
            out_filename = out_filename_base + '_' + stain + '.csv'           
            print(res_prop_diffs)
            print(len(res_prop_diffs))
            print(len(res_prop_diffs[0]))
            res_prop_diffs = list(map(list, zip_longest(*res_prop_diffs)))
            res_inner_diffs = list(map(list, zip_longest(*res_inner_diffs)))
            res_total_diffs = list(map(list, zip_longest(*res_total_diffs)))
            res_prop_average_diffs = list(map(list, zip_longest(*res_prop_average_diffs)))
            res_inner_average_diffs = list(map(list, zip_longest(*res_inner_average_diffs)))
            res_total_average_diffs = list(map(list, zip_longest(*res_total_average_diffs)))
            print(len(res_prop_diffs))
            print(len(res_prop_diffs[0]))
            # need to output csv in format ready for estimation stats.
            with open(out_filename, 'w') as f:
                print(res_prop_diffs)
                f.write('prop diffs:\n')
                f.write('2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene\n')
                for i in range(len(res_prop_diffs)):
                    for j in range(len(res_prop_diffs[i])):
                        f.write(str(res_prop_diffs[i][j])+', ')
                    f.write('\n')
                f.write('\n')
                print(res_inner_diffs)
                f.write('inner diffs:\n')
                f.write('2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene\n')
                for i in range(len(res_inner_diffs)):
                    for j in range(len(res_inner_diffs[i])):
                        f.write(str(res_inner_diffs[i][j])+', ')
                    f.write('\n')
                f.write('\n')
                print(res_total_diffs)
                f.write('total diffs:\n')
                f.write('2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene\n')
                for i in range(len(res_total_diffs)):
                    for j in range(len(res_total_diffs[i])):
                        f.write(str(res_total_diffs[i][j])+', ')
                    f.write('\n')
                f.write('\n')
                print(res_prop_average_diffs)
                f.write('average prop diffs:\n')
                f.write('2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene\n')
                for i in range(len(res_prop_average_diffs)):
                    for j in range(len(res_prop_average_diffs[i])):
                        f.write(str(res_prop_average_diffs[i][j])+', ')
                    f.write('\n')
                f.write('\n')
                print(res_inner_average_diffs)
                f.write('average inner diffs:\n')
                f.write('2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene\n')
                for i in range(len(res_inner_average_diffs)):
                    for j in range(len(res_inner_average_diffs[i])):
                        f.write(str(res_inner_average_diffs[i][j])+', ')
                    f.write('\n')
                f.write('\n')
                print(res_total_average_diffs)
                f.write('average total diffs:\n')
                f.write('2 wk dummy, 2 wk graphene, 6 wk dummy, 6 wk graphene, 12 wk dummy, 12 wk graphene\n')
                for i in range(len(res_total_average_diffs)):
                    for j in range(len(res_total_average_diffs[i])):
                        f.write(str(res_total_average_diffs[i][j])+', ')
                    f.write('\n')
                f.write('\n')

            
m1 = Metric('inner_intensity', inner_intensity)
m2 = Metric('prop', prop)
metrics = [m1, m2]
compare_sides = CompareSides('probe_side', 'contralateral', ['GFAP', 'IBA1'], ['2weekDummy', '2weekGraphene', '6weekDummy', '6weekGraphene', '12weekDummy', '12weekGraphene'],[m1, m2])
compare_sides.run()
compare_sides.perform_comparison()
compare_sides.output_result('test')

# now need to do same for neun.... needs to be a better structure to support multiple metrics





