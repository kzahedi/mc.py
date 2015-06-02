import numpy
import btk
import os
import math
import re
from random import random

parent   = "/home/somebody/projects/20141104-rbohand2/"
filename = parent + "grasp6_1-RBOHand2_2.trb"

###########################################################################
#                           read data functions                           #
###########################################################################

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

def walk(_dir, pattern, method):
    _dir = os.path.abspath(_dir)
    for _file in [_file for _file in os.listdir(_dir) if not _file in [".", "..", ".svn", ".git"]]:
        nfile = os.path.join(_dir, _file)
        if pattern.search(nfile):
            method(nfile)
        if os.path.isdir(nfile):
            walk(nfile, pattern, method)
                 

def get_domains_for_all_files(directory):
    pattern = re.compile(r".*RBOHand.*.trb$")
    files = []
    walk(directory, pattern, files.append)
    domains = [get_domain_for_each_marker(get_positions(filename)) for filename in files]
    global_domains = {}
    keys = domains[0].keys()
    for key in keys:
        domain = []
        for d in domains:
            minimum = d[key][0]
            maximum = d[key][1]
            if domain == []:
                domain = [ [minimum[0], minimum[1], minimum[2]], [maximum[0], maximum[1], maximum[2]]]

            d_min = [ domain[0][i] if minimum[i] > domain[0][i] else minimum[i] for i in range(0,3)]
            d_max = [ domain[1][i] if maximum[i] < domain[1][i] else maximum[i] for i in range(0,3)]
        global_domains[key] = [d_min, d_max]
    return global_domains

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

###########################################################################
#                            binning functions                            #
###########################################################################

def bin_value(v, bins):
  return min(int(v * bins), bins-1)

def bin_vector(v, bins):
  return [min(int(v[i] * bins), bins-1) for i in range(0,3)]

def bin_scaled_data_for_each_marker(data, bins):
  new_data = {}
  for key in data.keys():
    new_data[key] = [bin_vector(v, bins) for v in data[key]]
  return new_data

def combine_bin_vector(v, bins):
    return sum([v[i] * pow(bins,i) for i in range(0, len(v))])

def unique_valued_list(lst):
    myset = set(lst)
    return list(myset)

def relabel_vector(lst):
    mylst = unique_valued_list(lst)
    return [mylst.index(v) for v in lst]

def combine_bins_for_each_marker(data, bins):
    new_data = {}
    for key in data.keys():
        new_data[key] = relabel_vector([combine_bin_vector(v, bins) for v in data[key]])
    return new_data

def combine_random_variables(lst_of_lsts, bins):
    return relabel_vector([combine_bin_vector([v[i] for v in lst_of_lsts], bins) for i in range(0, len(lst_of_lsts[1]))])
    
###########################################################################
#                   calculating probabilities from data                   #
###########################################################################

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

###########################################################################
#                           MC quantifications                            #
###########################################################################

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

###########################################################################
#                 unique information based quantification                 #
###########################################################################

epsilon = 0.000001

def ml_eqeq(value, l): # matlab == operator
  return numpy.array([1 if value == x else 0 for x in numpy.ravel(l)])

def ml_lesser_than_zero(l):
  return numpy.array([1 if x < 0.0  else 0 for x in numpy.ravel(l)])

def ml_greater_than_zero(l):
  return numpy.array([1 if x > 0.0  else 0 for x in numpy.ravel(l)])

def ml_geq_zero(l):
  return numpy.array([1 if x >= 0.0 else 0 for x in numpy.ravel(l)])

def ml_extract(l, idx):
  r = numpy.empty(0)
  for index in range(0, len(l)):
      if idx[index] == 1:
          r = numpy.concatenate((r, [l[index]]))
  return r

def calculate_bases(nr_of_world_states, nr_of_action_states): # tested
  r = []
  x_range = range(nr_of_world_states)
  y_range = range(nr_of_world_states)
  z_range = range(nr_of_action_states)
  for x in range(0, nr_of_world_states):
    for y in range(1, nr_of_world_states):
      for z in range(1, nr_of_action_states):
        yp    = 0
        zp    = 0
        xe    = ml_eqeq(x,  x_range)
        ye    = ml_eqeq(y,  y_range)
        ype   = ml_eqeq(yp, y_range)
        ze    = ml_eqeq(z,  z_range)
        zpe   = ml_eqeq(zp, z_range)

        dxyz   = numpy.kron(ze,  numpy.kron(ye,  xe))
        dxypzp = numpy.kron(zpe, numpy.kron(ype, xe))
        dxypz  = numpy.kron(ze,  numpy.kron(ype, xe))
        dxyzp  = numpy.kron(zpe, numpy.kron(ye,  xe))
        r.append(dxyz + dxypzp - dxypz - dxyzp)

  return numpy.asarray(r)

def sample_from_delta_p(_p, _resolution):
  _nr_of_world_states = p.shape[0]
  # assert if shape[0] != shape[1]
  _nr_of_action_states = p.shape[2]
  _ps = numpy.ravel(_p)
  _dimension_of_delta_p = _nr_of_world_states * (_nr_of_world_states - 1) * (_nr_of_action_states - 1)
  _lst = [_ps] # list of return values
  _s   = 0

  _d = calculate_bases(_nr_of_world_states, _nr_of_action_states)

  _d = numpy.asmatrix(_d)

  for _ in range(0, _resolution):
    _a  = numpy.random.randn(_dimension_of_delta_p)
    _na = numpy.linalg.norm(_a)
    _a  = _a / _na
    _a  = numpy.asmatrix(_a)

    _b = _a * _d
    if numpy.count_nonzero(ml_eqeq(0, _b)) == 0:
      _v  = numpy.add(_ps, _s)
      _v  = -_v
      _v  = _v / _b
      _v  = numpy.ravel(_v)
      _vs = numpy.sign(_b)
      _vs = numpy.ravel(_vs)

      _vl = ml_lesser_than_zero(_vs)
      _vg = ml_greater_than_zero(_vs)

      _vvl = ml_extract(_v, _vl)
      _vvg = ml_extract(_v, _vg)

      _tmax = min(_vvl)
      _tmin = max(_vvg)

      _trand = numpy.random.rand(1)

      _to = _trand * (_tmax - _tmin) + _tmin
     
      _tob = _to * _a * _d
      _sd = _s + _tob
     
      _sdp = numpy.add(_sd, _ps)

      _gsdp = ml_geq_zero(_sdp)

      _pgsdp = numpy.prod(_gsdp)

      if _pgsdp >= 1.0:
        _s = _sd

        _sp = _s + _ps
        _lst.append(_sp)

  rlst = [numpy.reshape(numpy.ravel(elem),(_nr_of_world_states, _nr_of_world_states, _nr_of_action_states)) for elem in _lst]
  return rlst


def entropy(_distribution):
  return -sum([0 if v == 0 else v * math.log2(v) for v in numpy.ravel(_distribution)])

# MI(W';W|A) = H(W',A) + H(W,A) - H(W',W,A) - H(A)
# MI(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
def mi_xygz(_distribution):
  _xz = numpy.sum(_distribution, 1)
  _yz = numpy.sum(_distribution, 0)
  _z  = numpy.sum(_distribution, (0,1))
  return entropy(_xz) + entropy(_yz) - entropy(_distribution) - entropy(_z)

# MI(X;Z|Y) = H(X,Y) + H(Y,Z) - H(X,Y,Z) - H(Y)
# MI(W';A|W) = H(W',W) + H(W,A) - H(W',W,A) - H(W)
def mi_xzgy(_distribution):
  _xy = numpy.sum(_distribution, 2)
  _yz = numpy.sum(_distribution, 0)
  _y  = numpy.sum(_distribution, (0,2))
  return entropy(_xy) + entropy(_yz) - entropy(_distribution) - entropy(_y)

def mi_xy(_xy):
  _x  = numpy.sum(_xy, 0)
  _y  = numpy.sum(_xy, 1)
  _r = 0
  for _x_index in range(0, len(_x)):
    for _y_index in range(0, len(_y)):
      if _xy[_x_index][_y_index] > 0.0 and _x[_x_index] > 0.0 and _y[_y_index] > 0.0:
        _r = _r + _xy[_x_index][_y_index] * (math.log2(_xy[_x_index][_y_index])
             - math.log2(_x[_x_index] * math.log2(_y[_y_index])))
  return _r

# CoI(X;Y;Z) = MI(X;Y) - MI(X;Y|Z)
def coinformation(_distribution):
  _xy      = numpy.sum(_distribution, 2)
  _mi_xy   = mi_xy(_xy)
  _mi_xygz = mi_xygz(_distribution)
  return _mi_xy - _mi_xygz

def information_decomposition(_joint_distribution, _resolution):
  _samples = sample_from_delta_p(_joint_distribution, _resolution)
  _mi_xygz = [mi_xygz(_p) for _p in _samples]
  _mi_xzgy = [mi_xzgy(_p) for _p in _samples]
  _coiD    = coinformation(_joint_distribution)
  _coi     = [coinformation(_p) for _p in _samples]

  _synergistic   = min([v - _coiD for v in _coi])
  _uniqueWPrimeW = min(_mi_xygz)
  _uniqueWPrimeA = min(_mi_xzgy)
  
  return _synergistic, _uniqueWPrimeA, _uniqueWPrimeW



    
###########################################################################
#                            analyse functions                            #
###########################################################################

def analyse_directory(parent, nr_of_bins, functions):
    print "reading all files and looking for their domains"
    domains     = get_domains_for_all_files(parent)
    
    binned_actions = None
    
    pattern = re.compile(r".*RBOHand.*.trb$")
    files = []
    walk(directory, pattern, files.append)
    
    print "Only using the first three files for test reasons."
    files = files[0:3]
    
    results = {}
    
    for f in files:
        print "reading file " + f
        data = get_positions(f)
        print "scaling data"
        scaled_data = scale_data_for_each_marker(data, domains)
        print "binning data"
        binned_data = bin_scaled_data_for_each_marker(scaled_data, nr_of_bins)
        print "combining data"
        combined_binned_data = combine_bins_for_each_marker(binned_data, nr_of_bins)
        combined_binned_data = combine_random_variables([combined_binned_data[key] for key in combined_binned_data.keys()], nr_of_bins)
        if binned_actions == None:
            print "randomising action data"
            binned_actions = [int(random() * nr_of_bins) for v in range(1,len(combined_binned_data))]
        print "calculate joint distribution"
        jd = emperical_joint_distribution(combined_binned_data[2:len(combined_binned_data)], combined_binned_data[1:len(combined_binned_data)-1], binned_actions)
        
        r = {}
        for key in functions.keys():
            print "using method: " + key
            r[key] = functions[key](jd)
        results[f] = r
    print "done."
    return results
    
    
directory = "/home/somebody/projects/20141104-rbohand2/"
bins      = 100
functions = {"One" : calculate_concept_one, "Two" : calculate_concept_two}

r = analyse_directory(directory, bins, functions)


