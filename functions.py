import btk
import numpy
import os
import math

def bin_value(v, bins):
  return min(int(v * bins), bins-1)

def bin_vector(v, bins):
  return [min(int(v[i] * bins), bins-1) for i in range(0,3)]

def combine_bin_vector(v, bins):
  return sum([v[i] * pow(bins,i) for i in range(0, len(v))])

def unique_valued_list(lst):
  myset = set(lst)
  return list(myset)

def relabel_vector(lst):
  mylst = unique_valued_list(lst)
  return [mylst.index(v) for v in lst]


def get_positions(filename):
    reader = btk.btkAcquisitionFileReader() # build a btk reader object
    reader.SetFilename(filename) # set a filename to the reader
    reader.Update()
    acq = reader.GetOutput()
    
    frame_max =  acq.GetPointFrameNumber() #get the number of frames

    all_points = []
    for j in range(0, frame_max):
      points = {}
      for i in range(acq.GetPointNumber()):
        point = acq.GetPoint(i)
        label = point.GetLabel()
        values = point.GetValues() #returns a numpy array of the data, row=frame, col=coordinates
        points[label] = values[j][:]
        all_points.append(points)

    return all_points

def get_domain_for_each_marker(data):
  domains = {}
  for key in data[0].keys():
    min_values = [data[0][key][i] for i in range(0,3)]
    max_values = [data[0][key][i] for i in range(0,3)]
    for j in range(1, len(data)):
      values = data[j][key]
      for i in range(0,3):
        if values[i] < min_values[i]:
          min_values[i] = values[i]
        if values[i] > max_values[i]:
          max_values[i] = values[i]
    domains[key] = [min_values, max_values]
  return domains

def scale_data_for_each_marker(data, domains):
  scaled_data = {}
  for key in domains.keys():
    new_data = []
    min_values = domains[key][0]
    max_values = domains[key][1]
    for j in range(0, len(data)):
      values = data[j][key]
      # if values[0] != nan:
      if max([abs((max_values[i] - min_values[i])) for i in range(0,3)]) > 0.00001:
        values = [(values[i] - min_values[i]) / (max_values[i] - min_values[i])
              for i in range(0,3)]
      new_data.append(values)
    scaled_data[key] = new_data
  return scaled_data

def bin_scaled_data_for_each_marker(data, bins):
  new_data = {}
  for key in data.keys():
    new_data[key] = [bin_vector(v, bins) for v in data[key]]
  return new_data

def combine_bins_for_each_marker(data, bins):
  new_data = {}
  for key in data.keys():
    new_data[key] = relabel_vector([combine_bin_vector(v, bins) for v in data[key]])
  return new_data

def emperical_joint_distribution(w_prime, w, a):
  p = numpy.zeros((max(w_prime)+1, max(w)+1, max(a)+1))

  L = len(w_prime)
  for index in range(0, L):
    p[w_prime[index], w[index], a[index]] = p[w_prime[index], w[index], a[index]] + 1.0

  for i in range(0, p.shape[0]):
    for j in range(0, p.shape[1]):
      for k in range(0, p.shape[2]):
        p[i,j,k] = p[i,j,k] / float(L)

  s = sum(sum(sum(p)))
  p = p / s
  return p

def calc_p_w_prime_given_w(joint_distribution):
    p_w_prime_w = joint_distribution.sum(axis=2)
    p_w         = joint_distribution.sum(axis=(0,2))
    for w_prime in range(0,joint_distribution.shape[0]):
        for w in range(0, joint_distribution.shape[1]):
            p_w_prime_w[w_prime, w] = p_w_prime_w[w_prime, w] / p_w[w]
    return p_w_prime_w

def calc_p_w_prime_given_a(joint_distribution):
    p_w_prime_a = joint_distribution.sum(axis=1)
    p_a         = joint_distribution.sum(axis=(0,1))
    for w_prime in range(0,joint_distribution.shape[0]):
        for a in range(0, joint_distribution.shape[2]):
            if p_w_prime_a[w_prime, a] != 0.0 and p_a[a] != 0.0:
                p_w_prime_a[w_prime, a] = p_w_prime_a[w_prime, a] / p_a[a]
    return p_w_prime_a

def calc_p_w_prime_given_w_a(joint_distribution):
    p_w_a               = joint_distribution.sum(axis=0)
    p_w_prime_given_w_a = numpy.zeros(joint_distribution.shape)
    for w_prime in range(0, joint_distribution.shape[0]):
        for w in range(0, joint_distribution.shape[1]):
            for a in range(0, joint_distribution.shape[2]):
                if joint_distribution[w_prime, w, a] != 0.0 and p_w_a[w,a] != 0.0:
                    p_w_prime_given_w_a[w_prime, w, a] = joint_distribution[w_prime, w, a] / p_w_a[w,a]
    return p_w_prime_given_w_a

def calculate_concept_one(joint_distribution):
    p_w_prime_given_w   = calc_p_w_prime_given_w(joint_distribution)
    p_w_prime_given_w_a = calc_p_w_prime_given_w_a(joint_distribution)
    r = 0
    for w_prime in range(0, joint_distribution.shape[0]):
        for w in range(0, joint_distribution.shape[1]):
            for a in range(0, joint_distribution.shape[2]):
                if joint_distribution[w_prime, w, a] != 0.0 and p_w_prime_given_w[w_prime, w] != 0.0 and p_w_prime_given_w_a[w_prime, w, a] != 0.0:
                    r = joint_distribution[w_prime, w, a] * (math.log(p_w_prime_given_w_a[w_prime, w, a], 2) - math.log(p_w_prime_given_w[w_prime, w], 2))
    return r

def calculate_concept_two(joint_distribution):
    p_w_prime_given_a   = calc_p_w_prime_given_a(joint_distribution)
    p_w_prime_given_w_a = calc_p_w_prime_given_w_a(joint_distribution)
    r = 0
    for w_prime in range(0, joint_distribution.shape[0]):
        for w in range(0, joint_distribution.shape[1]):
            for a in range(0, joint_distribution.shape[2]):
                if joint_distribution[w_prime, w, a] != 0.0 and p_w_prime_given_a[w_prime, a] != 0.0 and p_w_prime_given_w_a[w_prime, w, a] != 0.0:
                    r = joint_distribution[w_prime, w, a] * (math.log(p_w_prime_given_w_a[w_prime, w, a], 2) - math.log(p_w_prime_given_a[w_prime, a], 2))
    return r
