#!/usr/bin/env python

import os
import sys
import csv
import yaml
import re
from itertools import product
import math
from random import shuffle
from mathematics import is_number
import numpy as np
from scipy import optimize
from element_data import atomic_number
from element_data import element_location
from atoms import Structure
from composition import Composition
from composition import sort_systems
from kpoints import Kmesh
from incar import Incar
from inputs import inputs
from jobscript import Jobscript
from mysub import cd, execute_local, execute_remote

# gas = []
gas = ['H', 'N', 'O', 'F', 'Cl', 'Xe']
liquid = ['Br', 'Hg']

guess_lengths = {'H':0.74, 'N':1.10, 'O':1.21, 'F':1.42, 'Cl':1.99, 'Br':2.28}
# from Tables of Interatomic Distances and Configuration of Molecules
# and Ions, Special Publication No 11; Supplement 1956-1959, Special
# Publication No 18, Chemical Society, London, 1958, 1965, accessed
# from http://www.kayelaby.npl.co.uk/chemistry/3_7/3_7_2.html

magnetic_3d = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']
magnetic_4d = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag']
magnetic_5d = ['Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au']
magnetic_4f = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb']
magnetic_5f = ['Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No']
magnetic_els = magnetic_3d + magnetic_4d + magnetic_5d + magnetic_4f + magnetic_5f

known_clusters = ['cypress', 'edison']
cores_per_node = {'cypress': 28, 'edison': 24}
down_clusters = []

def for_each_calculation(function):
    def wrapper(self, **kwargs):
        fxn_kwargs = kwargs.copy() ; del fxn_kwargs['funcs'] ; del fxn_kwargs['jobtypes']
        for system in self.systems:
            for func in kwargs['funcs']:
                for jobtype in kwargs['jobtypes']:
                    folder = os.path.join(self.basedir, 'dft_runs', system, func, jobtype)
                    fxn_kwargs['system'] = system
                    fxn_kwargs['func'] = func
                    fxn_kwargs['jobtype'] = jobtype
                    with cd(folder):
                        function(self, **fxn_kwargs)
    return wrapper

def gp_name(compound):
    '''Generate a gnuplot-friendly compound string

    This will get the subscripts correctly. Also, since we use the
    Composition object it will use our conventional element ordering

    '''
    return Composition(compound).format_string('gnuplot')

def oqmd_dftu(compound):
    '''Returns True if compound treated with DFT+U in OQMD'''
    els = Composition(compound).elements()
    corr_els = ['V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Th', 'U', 'Np', 'Pu']
    if 'O' in els and any([el in els for el in corr_els]):
        # must contain (1) O and (2) actinides and/or certain TM
        print "Ignoring %s since treated with DFT+U in OQMD"%compound
        return True
    else:
        return False

def grab_poscar(entryid, fcc_length=20., symb=''):
    '''Grab the qmpy poscar file for a given entry id

    guess diatomic molecule POSCAR for entryid of -2

    guess atom POSCAR for entryid of -1

    '''
    if entryid == -2:
        # diatomic gas
        dist = guess_lengths[symb]
        pos_template = '''\
%s
1.0
%s %s  0
 0 %s %s
%s  0 %s
%s
2
Cartesian
0 0 0
0 0 %.4f
'''
        mol = Structure(pos_template%(symb, fcc_length/2., fcc_length/2., fcc_length/2., fcc_length/2., fcc_length/2., fcc_length/2., symb, dist))
        return mol.pos2vasp(ret=True)
    elif entryid == -1:
        # monatomic gas
        pos_template = '''\
%s
1.0
%s %s  0
 0 %s %s
%s  0 %s
%s
1
Cartesian
0 0 0
'''
        mol = Structure(pos_template%(symb, fcc_length/2., fcc_length/2., fcc_length/2., fcc_length/2., fcc_length/2., fcc_length/2., symb))
        return mol.pos2vasp(ret=True)
    else:
        # solid
        return execute_remote('larue', "/home/mko570/scripts/entry2poscar.sh %i"%entryid)[0]

def generate_jobscript(cluster, params={'jobtype': 'vasp_scan', 'nnodes': 1, 'nhours': 2}, out='run.vasp'):
    '''Generate job script

    cluster:   cluster to run on
    params:    dict of settings fed to Jobscript.set_params
    out:       name of job script

    '''
    # print job script
    js = Jobscript(cluster)
    if 'email' not in params:
        params['email'] = 'mkothako@tulane.edu' if cluster == 'cypress' else 'manish.hcu.phy@gmail.com'
    # make sure walltime not too large
    if 'nhours' in params:
        if cluster == 'edison' and params['nhours'] > 36:
            params['nhours'] = 36
        if cluster == 'cypress' and params['nhours'] > 168:
            params['nhours'] = 168
    js.set_params(params)
    js.gen_jobscript()
    js.print_jobscript(out)

    # npar = round(sqrt(total_ncores)) rounded to power of two
    if os.path.isfile('POSCAR'):
        pos = Structure('POSCAR')
    elif os.path.isfile('../relax/POSCAR'):
        pos = Structure('../relax/POSCAR')
    else: sys.exit('Error: Cannot find POSCAR in %s directory'%os.getcwd())
    inc = Incar(pos)
    inc.read_incar()
    inc.incar['NPAR'] = int(2**(round(math.log(np.sqrt(js.nnodes*cores_per_node[cluster]), 2))))
    inc.print_incar('INCAR')

def cleanup_vasp(folder='.', remove_extras=True, archive=True, keep=[]):
    '''Delete all VASP files except inputs

    remove_extras:   also remove the important output files
    archive:         save a copy of vasp.out
    keep:            list of files to not delete

    '''
    always = ['CHG', 'CHGCAR', 'CONTCAR', 'DOSCAR', 'EIGENVAL', 'IBZKPT', 'PCDAT', 'PROCAR', 'REPORT', 'vasprun.xml', 'WAVECAR', 'XDATCAR']
    extras = ['DYNMAT', 'job.err', 'job.out', 'OSZICAR', 'OUTCAR', 'PID', 'vasp.job', 'vasp.out', 'dm', 'dos', 'pdos']
    with cd(folder):
        if archive:
            if os.path.isfile('vasp.out'):
                # make sure to not overwrite previous archives of vasp.out
                check_files = execute_local("ls vasp.out.* 2>/dev/null | sed 's#vasp.out.##g' | sort -n | tail -1")[0].splitlines()
                suffix = int(float(check_files[0])) + 1 if len(check_files) > 0 else 1
                execute_local("cp vasp.out vasp.out.%i"%suffix)
        files_to_remove = ''
        files_to_remove += ' '.join([fil for fil in always if fil not in keep])
        if remove_extras: files_to_remove += ' ' + ' '.join([fil for fil in extras if fil not in keep])
        execute_local("rm -rf %s"%files_to_remove)

def add_resources(cluster, folder='.', filename='run.vasp'):
    '''Add resources to a job script'''
    with cd(folder):
        if os.path.isfile(filename):
            # Determine previous nodes and walltime
            js = Jobscript(cluster)
            js.read_jobscript(infile=filename)
            if hasattr(js, 'ppn'): delattr(js, 'ppn')
            nnodes = js.nnodes ; nhours = js.nhours
            new_resources = True
            if cluster == 'cypress':
              # if nhours < 84:
                  # 2x hours unless that will put us over cypress limit of 7 days
                  # nhours *= 2
              # elif nhours < 150:
              if nhours < 150:
                  # bump up to max hours if still not at max
                  nhours = 168
              elif nnodes < 2:
                  # 2x nodes unless that will put us over 2 nodes
                  nnodes *= 2
              else: new_resources = False
            elif cluster == 'edison':
              if nhours < 22:
                  # bump up to max hours if still not at max
                  nhours = 36
              elif nnodes < 4:
                  # 2x nodes unless that will put us over 6 nodes
                  nnodes *= 2
              else: new_resources = False
            else: sys.exit('Error: cluster %s is not implemented'%cluster)
            generate_jobscript(cluster, {'jobtype': 'vasp_scan', 'nnodes': nnodes, 'nhours': nhours}, out=filename)
            return new_resources
        else: sys.exit('Error: Cannot find %s in %s directory'%(filename, os.getcwd()))

def check_job(folder, job_status):
    '''Check if job is complete and, if so, fetch it and return True'''
    ready, submitted, queued, running, cluster, jobid, processed = job_status[folder]
    if cluster in down_clusters: print 'ignoring since %s cluster is down'%cluster ; return False
    if ready:
        if submitted:
            if not queued:
                if not running:
                    if not processed:
                        with cd(folder):
                            # bring back to local machine
                            execute_local("/home/mkothako/scripts/fetchjob.sh %s %s > /dev/null"%(cluster, 'myscratch' if cluster == 'cypress' else ''))
                        return True
                    elif processed:
                        print "already processed" ; return False
            else:
                if not running and not processed:
                    print "in queue" ; return False
                elif running and not processed:
                    print "running" ; return False
        else:
            if not queued and not running and not processed:
                print "not yet submitted" ; return False
    else:
        if not submitted and not queued and not running and not processed:
            print "not ready" ; return False

def fix_vasp_errors(folder):
    '''Try to fix any VASP errors for an unconverged calculation'''
    if os.path.isfile(os.path.join(folder, 'vasp.out')):
        # find error messages in vasp.out
        errors_found = []
        with open(os.path.join(folder, 'vasp.out'), 'r') as fp:
            for line in fp:
                if 'Sub-Space-Matrix is not hermitian in DAV' in line:
                    if 'subspace_matrix_hermitian' not in errors_found:
                        errors_found.append('subspace_matrix_hermitian')
                elif 'inverse of rotation matrix was not found' in line:
                    if 'inverse_rotation_matrix' not in errors_found:
                        errors_found.append('inverse_rotation_matrix')
                elif 'Found some non-integer element in rotation matrix' in line:
                    if 'non_integer_rotation_matrix' not in errors_found:
                        errors_found.append('non_integer_rotation_matrix')
                elif 'Your highest band is occupied at some k-points!' in line:
                    if 'too_few_bands' not in errors_found:
                        errors_found.append('too_few_bands')
        if 'relax' in folder:
            errors_found.append('loosen_ediffg')
            errors_found.append('lcharg_false') # not error but want to default on now
        # address errors
        posfile = 'CONTCAR' if os.path.isfile(os.path.join(folder, 'CONTCAR')) and os.path.getsize('CONTCAR') > 0 else 'POSCAR'
        if os.path.isfile(posfile):
            if 'subspace_matrix_hermitian' in errors_found:
                # set ALGO to Fast
                prev_incar = Incar(Structure(posfile))
                prev_incar.read_incar(os.path.join(folder, 'INCAR'))
                if 'IALGO' in prev_incar.incar: del prev_incar.incar['IALGO']
                prev_incar.incar['ALGO'] = 'Fast'
                prev_incar.print_incar('INCAR')
            if 'inverse_rotation_matrix' in errors_found:
                # use odd k-mesh
                km = Kmesh(posfile)
                km.read_kmesh()
                new_km = []
                for i in range(3):
                    if km.kmesh[i] % 2 == 0:
                        new_km.append(km.kmesh[i]+1)
                    else:
                        new_km.append(km.kmesh[i])
                km.kmesh = tuple(new_km)
                km.print_kmesh(out='KPOINTS')
                # turn off symmetry
                prev_incar = Incar(Structure(posfile))
                prev_incar.read_incar(os.path.join(folder, 'INCAR'))
                prev_incar.incar['ISYM'] = 0
                prev_incar.print_incar('INCAR')
            if 'non_integer_rotation_matrix' in errors_found:
                # turn off symmetry
                prev_incar = Incar(Structure(posfile))
                prev_incar.read_incar(os.path.join(folder, 'INCAR'))
                prev_incar.incar['ISYM'] = 0
                prev_incar.print_incar('INCAR')
            if 'loosen_ediffg' in errors_found:
                # loosen ediffg to -0.005
                prev_incar = Incar(Structure(posfile))
                prev_incar.read_incar(os.path.join(folder, 'INCAR'))
                if abs(float(prev_incar.incar['EDIFFG'])) < 0.005:
                    prev_incar.incar['EDIFFG'] = -0.005
                    prev_incar.print_incar('INCAR')
            if 'lcharg_false' in errors_found:
                # turn on lcharg
                prev_incar = Incar(Structure(posfile))
                prev_incar.read_incar(os.path.join(folder, 'INCAR'))
                prev_incar.incar['LCHARG'] = '.TRUE.'
                prev_incar.print_incar('INCAR')
            if 'too_few_bands' in errors_found:
                # increase nbands by larger of 25% or 2 bands
                prev_incar = Incar(Structure(posfile))
                prev_incar.read_incar(os.path.join(folder, 'INCAR'))
                if 'NBANDS' in prev_incar.incar:
                    prev_incar.incar['NBANDS'] = int(max(prev_incar.incar['NBANDS']+2, 1.25*prev_incar.incar['NBANDS']))
                else:
                    # grab NBANDS from OUTCAR
                    prev_nbands = int(float(execute_local('grep NBANDS= OUTCAR')[0].split()[-1]))
                    prev_incar.incar['NBANDS'] = int(max(prev_nbands+2, 1.25*prev_nbands))
                prev_incar.print_incar('INCAR')
        else: sys.exit('Error: Cannot find structure file %s for unconverged run'%posfile)
    else: sys.exit('Error: Cannot find vasp.out file for unconverged run')


class Benchmark:
    '''Class to generate and analyze scan calculations

    basedir: directory for calculations

    '''

    def __init__(self, basedir='.'):
        '''Initialize the benchmark'''
        # calculations location
        self.basedir = basedir
        # create base directory if not there
        if not os.path.isdir(self.basedir): os.makedirs(self.basedir)
        # dict of compound: entry
        self.compounds = {}
        # same but for all solid elements
        self.all_solid_elements = {}
        # same but only for solid elements needed
        self.solid_elements = {}
        # for cases in which reference phase is not solid, we use a
        # single molecule, e.g. O_2 for an oxide
        self.non_solid_elements = []
        # list of the systems (compounds and elements)
        self.systems = []
        # dict of system: POSCAR string (just initial POSCAR)
        self.poscars = {}
        # dict of system: experimental POSCAR string (no gases)
        self.exp_poscars = {}
        # dict of calculation settings
        self.calc_settings = {}
        # formation energy dict, from outside to inside...
        # (1) 'calc' or 'exp'
        # (2) 'pbe' or 'scan' for 'calc', 'iit' or 'ssub' for 'exp'
        # (3) compound: fe
        # for experiments, we have a list of formation energies in case of multiple
        self.fe = {}
        # zero-point energy dict, from outside to inside...
        # (1) element (e.g. 'H', 'N', etc.)
        # (2) 'calc' or 'exp'
        # (3) 'pbe' or 'scan' for 'calc', 'irikura' for 'exp'
        self.zpe = {}
        # property dicts, from outside to inside...
        # (1) system (e.g. CoO, Fe)
        # (2) func ('pbe' or 'scan')
        # (3) property
        self.volume = {} # volume per atom
        self.exp_volume = {} # experimental volume per atom
        self.moms = {} # list of magnetic moments (mu_B)
        self.gaps = {} # band gaps (eV)
        self.exp_gaps = {} # experimental band gaps

    def compounds_from_file(self, filename, elreference=False):
        '''Generate the list of compounds from a QmpyQuery file

        filename:      csv QmpyQuery file containing the compounds
        elreference:   True if reading in elemental references

        '''
        print "[Reading %s list from %s]"%('element' if elreference else 'compound', filename)
        # Load in the QmpyQuery csv result
        if not os.path.isfile(os.path.join(self.basedir, filename)): sys.exit('Error: Cannot find %s file'%filename)
        qmpyquery = open(os.path.join(self.basedir, filename), 'r')
        reader = csv.reader(qmpyquery)
        if elreference:
            # solid reference states
            for row in reader:
                if row[0] != 'Composition':
                    self.all_solid_elements[row[0]] = int(float(row[1]))
        else:
            # compounds
            for row in reader:
                if row[0] != 'Composition':
                    self.compounds[row[0]] = int(float(row[1]))

    def print_compounds(self):
        '''Print out the compound list'''
        if len(self.compounds) > 0:
            print "%s"%('-'*30)
            print "%18s %10s"%('Compound', 'Entry ID')
            print "%s"%('-'*30)
        for compound in sort_systems(self.compounds):
            print "%18s %10i"%(compound, self.compounds[compound])
        if len(self.compounds) > 0:
            print "%s"%('-'*30)

    def save_compounds(self, outfile='compounds.csv'):
        '''Save the compound list to outfile'''
        print "[Saving compound list as %s]"%outfile
        if len(self.compounds) > 0:
            table = [[key, self.compounds[key]] for key in sort_systems(self.compounds)]
            with open(os.path.join(self.basedir, outfile), 'w') as f:
                writer = csv.writer(f)
                writer.writerows([['Composition', 'Entry ID']])
                writer.writerows(table)

    def filter_compounds(self, max_natoms=999, contains=[''], lacks=[''], exp_fe=[''], remove_ambiguous_exp_fe=False, all_calcs_finished=False, funcs=['pbe', 'scan']):
        '''Filter the set of compounds

        This can enable a consistent analysis even while calculations
        (e.g. of compounds with larger unit cells) are still ongoing

        Also it enables analysis of specific subsets of compounds

        max_natoms: maximum number of atoms in the unit cell

        contains: list of chemistry requirements (all must be
        satisfied)

        each list member is a string containing comma-separated
        element symbols, one or more of which must be contained by the
        candidate compound (e.g. 'Fe,Co,Ni')

        lacks: list of forbidden elements (e.g. ['O', 'S', 'Se'])

        if remove_ambiguous_exp_fe, compounds for which any pair of
        exp fe values varies by over 100% are removed (before any
        exp_fe filtering)

        exp_fe: list of experimental f.e. requirements (all must be
        satisfied)

        each list member is a string containing an inequality (e.g.
        '<-1.0' or '>0.0'); square brackets for abs value

        the exp. f.e. is taken to be the average over different
        sources if multiple (similarly, within each source an average
        is taken if there are multiple values)

        all_calcs_finished: if True, ignore a compound if we are
        missing any functional

        '''
        funcs = self._set_funcs(funcs)
        print 'Starting with %i compounds'%len(self.compounds)
        if max_natoms != 999:
            # filter by max no of atoms
            print "[Filtering compounds by natoms <= %i]"%max_natoms
            to_remove = []
            for compound in self.compounds:
                pos = Structure(self.poscars[compound])
                if pos.totalnatoms > max_natoms:
                    to_remove.append(compound)
            self.compounds = {comp: self.compounds[comp] for comp in self.compounds if comp not in to_remove}
        if contains != ['']:
            # filter by chemistry (necessary elements)
            print "[Filtering compounds by chemistry: contains %s]"%(' & '.join(['|'.join(req.split(',')) for req in contains]))
            to_remove = []
            for compound in self.compounds:
                if not all(any(Composition(compound).contains(el) for el in req.split(',')) for req in contains):
                    to_remove.append(compound)
            self.compounds = {comp: self.compounds[comp] for comp in self.compounds if comp not in to_remove}
        if lacks != ['']:
            # filter by chemistry (forbidden elements)
            print "[Filtering compounds by chemistry: lacks %s]"%(' & '.join(lacks))
            to_remove = [compound for compound in self.compounds if any(Composition(compound).contains(el) for el in lacks)]
            self.compounds = {comp: self.compounds[comp] for comp in self.compounds if comp not in to_remove}
        if remove_ambiguous_exp_fe:
            # remove compounds with large discrepancy in exp f.e.
            print "[Removing compounds with any difference in experimental f.e. > 100%%]"
            to_remove = []
            for compound in self.compounds:
                exp_values = []
                if 'ssub' in self.fe[compound]['exp']: exp_values.extend(self.fe[compound]['exp']['ssub'])
                if 'iit' in self.fe[compound]['exp']: exp_values.extend(self.fe[compound]['exp']['iit'])
                if any([any([abs(100*(exp_val_2-exp_val_1)/exp_val_1) > 100. for exp_val_2 in exp_values]) for exp_val_1 in exp_values]):
                    to_remove.append(compound)
            if len(to_remove): print 'Removing %s'%(','.join(to_remove))
            self.compounds = {comp: self.compounds[comp] for comp in self.compounds if comp not in to_remove}
        if exp_fe != ['']:
            # filter by experimental formation energy
            def operation(symbol):
                op_dict = {'<': '<', '>': '>', '[': '<', ']': '>'}
                return op_dict[symbol]
            def use_abs_value(symbol):
                return symbol in ['[', ']']
            print "[Filtering compounds by experimental f.e. of %s]"%(' & '.join([req.replace('[', '|<').replace(']', '|>') for req in exp_fe]))
            # make sure no problems with conditions
            for req in exp_fe:
                if req[0] not in ['<', '>', '[', ']']:
                    sys.exit('Error: Cannot understand inequality symbol %s in exp f.e. filtering condition %s'%(req[0], req))
                elif not is_number(req[1:]):
                    sys.exit('Error: Cannot understand value %s in exp f.e. filtering condition %s'%(req[1:], req))
            to_remove = []
            for compound in self.compounds:
                # each condition is something like -2.34545 < -1.0
                # where the l.h.s. is the exp fe and r.h.s. comes from the condition
                exp_fe_val = np.mean([np.mean(self.fe[compound]['exp'][source]) for source in self.fe[compound]['exp']])
                if not all([eval('%s%s%s %s %s'%('abs(' if use_abs_value(req[0]) else '', exp_fe_val, ')' if use_abs_value(req[0]) else '', operation(req[0]), req[1:])) for req in exp_fe]):
                    to_remove.append(compound)
            if len(to_remove): print 'Removing %s'%(','.join(to_remove))
            self.compounds = {comp: self.compounds[comp] for comp in self.compounds if comp not in to_remove}
        if all_calcs_finished:
            # remove any compound for which not all the functional
            # calculations are complete
            print "[Filtering compounds by whether all calculations are finished]"
            to_remove = []
            folders = []
            for system in self.systems:
                for func in funcs:
                    folders.append(os.path.join(self.basedir, 'dft_runs', system, func, 'rel_geom'))
            job_statuses = self.job_statuses(folders, only_local=True)
            for compound in self.compounds:
                has_needed_calcs = []
                has_needed_calcs.append(all([job_statuses[os.path.join(self.basedir, 'dft_runs', compound, func, 'rel_geom')][6] for func in funcs]))
                cc = Composition(compound)
                for el in cc.elements():
                    has_needed_calcs.append(all([job_statuses[os.path.join(self.basedir, 'dft_runs', el, func, 'rel_geom')][6] for func in funcs]))
                if not all(has_needed_calcs):
                    to_remove.append(compound)
            if len(to_remove): print 'Removing %s'%(','.join(to_remove))
            self.compounds = {comp: self.compounds[comp] for comp in self.compounds if comp not in to_remove}
        self.fe = {comp: self.fe[comp] for comp in self.fe if comp in self.compounds} # update fe dict
        print 'Finishing with %i compounds'%len(self.compounds)

    def gen_elements(self, ref_file='reference_states.csv'):
        '''Generate the elements needed

        ref_file: QmpyQuery csv file for elemental references

        '''
        # Read in the solid reference states
        self.compounds_from_file(filename=ref_file, elreference=True)
        self.solid_elements = {} ; self.non_solid_elements = []
        for compound in self.compounds:
            for el in Composition(compound).elements():
                if el in gas or el in liquid:
                    if el in liquid: print "Warning: %s is a liquid at standard conditions!"%el
                    if el not in self.non_solid_elements: self.non_solid_elements.append(el)
                else:
                    if el not in self.solid_elements: self.solid_elements[el] = self.all_solid_elements[el]

    def print_elements(self):
        '''Print out the element list'''
        if len(self.solid_elements) == 0 and len(self.non_solid_elements) == 0:
            self.gen_elements()
        if len(self.compounds) > 0:
            print "%s"%('-'*30)
            print "%18s %10s"%('Element', 'Entry ID')
            print "%s"%('-'*30)
        for element in sort_systems(self.solid_elements):
            print "%18s %10i"%(element, self.solid_elements[element])
        for element in sort_systems(self.non_solid_elements):
            if element in ['Xe']:
                print "%18s %10s"%(element, 'monatomic')
            else:
                print "%18s %10s"%(element, 'diatomic')
        if len(self.solid_elements) > 0 or len(self.non_solid_elements):
            print "%s"%('-'*30)

    def save_elements(self, outfile='elements.csv'):
        '''Save the element list to outfile'''
        print "[Saving element list as %s]"%outfile
        if len(self.solid_elements) > 0 or len(self.non_solid_elements) > 0:
            table_solid = [[el, self.solid_elements[el]] for el in sort_systems(self.solid_elements)]
            table_nonsolid = [[el, 'monatomic' if el in ['Xe'] else 'diatomic'] for el in sort_systems(self.non_solid_elements)]
            with open(os.path.join(self.basedir, outfile), 'w') as f:
                writer = csv.writer(f)
                writer.writerows([['Element', 'Entry ID']])
                writer.writerows(table_solid)
                writer.writerows(table_nonsolid)

    def gen_systems(self, sort=True):
        '''Fill up systems list'''
        print "[Generating list of systems]"
        self.systems = []
        for system in self.solid_elements.keys() + self.non_solid_elements + self.compounds.keys():
            self.systems.append(system)
        if sort: self.systems = sort_systems(self.systems)

    def elemental_occurrence(self):
        '''Compute the occurrence of each element in the compound set'''
        print "[Computing elemental occurrences for the %i compounds]"%(len(self.compounds))
        occurrences = {}
        for compound in self.compounds:
            for el in Composition(compound).elements():
                if el not in occurrences:
                    occurrences[el] = 1
                else:
                    occurrences[el] += 1
        print '# Element Occurrences'
        for el in sort_systems(occurrences):
            print '%2s %4i'%(el, occurrences[el])

    def gen_poscars(self):
        '''Fill up poscars dictionary with POSCAR strings'''
        print "[Generating POSCAR files]"
        for system in self.systems:
            if system in self.compounds:
                self.poscars[system] = grab_poscar(self.compounds[system])
            elif system in self.solid_elements:
                self.poscars[system] = grab_poscar(self.solid_elements[system])
            elif system in self.non_solid_elements:
                if system in ['Xe']:
                    # -1 indicates single atom
                    self.poscars[system] = grab_poscar(-1, symb=system)
                else:
                    # -2 indicates diatomic molecule
                    self.poscars[system] = grab_poscar(-2, symb=system)
            else: sys.exit('Error: %s system is not a compound, solid element, or non solid element'%system)

    def grab_exp_poscars(self):
        '''Fill up exp_poscars dictionary with experimental POSCAR strings'''
        print "[Grabbing experimental POSCAR files]"
        for system in self.systems:
            if system in self.compounds:
                try:
                    self.exp_poscars[system] = execute_remote('larue', "/home/mko570/scripts/entry2expposcar.sh %i"%self.compounds[system])[0]
                except:
                    print 'Warning: Failed to grab experimental structure for %s'%system
            elif system in self.solid_elements:
                self.exp_poscars[system] = execute_remote('larue', "/home/mko570/scripts/entry2expposcar.sh %i"%self.solid_elements[system])[0]
            elif system in self.non_solid_elements:
                pass # ignore if gas
            else: sys.exit('Error: %s system is not a compound, solid element, or non solid element'%system)

    def save_poscars(self, outfile='poscars.csv', experimental=False):
        '''Save the poscars to outfile as csv

        The exp_poscars data is written instead of the poscars data if
        experimental is set to True

        '''
        print "[Saving %sPOSCAR files as %s]"%('experimental ' if experimental else '', outfile)
        w = csv.writer(open(os.path.join(self.basedir, outfile), 'w'))
        pos_items = self.exp_poscars.items() if experimental else self.poscars.items()
        for key, val in pos_items:
            w.writerow([key, repr(val)])

    def read_poscars(self, infile='poscars.csv', experimental=False):
        '''Read poscars from infile csv

        The exp_poscars dict is set instead of the poscars dict if
        experimental is set to True

        '''
        print "[Reading %sPOSCAR files from %s]"%('experimental ' if experimental else '', infile)
        reader = csv.reader(open(os.path.join(self.basedir, infile), 'r'))
        if experimental:
            for row in reader: self.exp_poscars[row[0]] = row[1].replace('\\n', '\n').replace("'", "")
        else:
            for row in reader: self.poscars[row[0]] = row[1].replace('\\n', '\n').replace("'", "")

    def visualize(self, funcs='', jobtype='auto', only_incomplete=True):
        '''Visualize the structures'''
        nav_options = '(y=yes, n=no, q=quit, #=jump, l=legend)'
        funcs = self._set_funcs(funcs)
        print "[Visualizing structures %s]"%nav_options
        structure_files = []
        for compound in self.systems:
            for func in funcs:
                if jobtype == 'relax':
                    jobtypes = [jobtype]
                elif jobtype == 'rel_geom':
                    jobtypes = [jobtype]
                elif jobtype == 'auto':
                    if os.path.isfile(os.path.join(self.basedir, 'dft_runs', compound, func, 'rel_geom', 'POSCAR')) or os.path.isfile(os.path.join(self.basedir, 'dft_runs', compound, func, 'rel_geom', 'CONTCAR')):
                        jobtypes = ['rel_geom']
                    elif os.path.isfile(os.path.join(self.basedir, 'dft_runs', compound, func, 'relax', 'POSCAR')) or os.path.isfile(os.path.join(self.basedir, 'dft_runs', compound, func, 'relax', 'CONTCAR')):
                        jobtypes = ['relax']
                    else:
                        jobtypes = []
                        print 'Warning: Cannot determine which jobtype to visualize structure for %s, %s'%(compound, func)
                for jbtyp in jobtypes:
                    folder = os.path.join(self.basedir, 'dft_runs', compound, func, jbtyp)
                    with cd(folder):
                        if os.path.isfile('CONTCAR'):
                            pos_filename = 'CONTCAR'
                        elif os.path.isfile('POSCAR'):
                            pos_filename = 'POSCAR'
                        else: sys.exit('Error: Cannot find CONTCAR or POSCAR in %s directory for visualization'%folder)
                        # visualize complete job only if only_incomplete is off
                        viz = not only_incomplete if pos_filename == 'CONTCAR' and jbtyp == 'rel_geom' else True
                        if viz:
                            structure_files.append([compound, func, jbtyp, pos_filename])
        pos_num = 0 ; tot_num = len(structure_files)
        while pos_num < tot_num:
            compound, func, jbtyp, pos_filename = structure_files[pos_num]
            folder = os.path.join(self.basedir, 'dft_runs', compound, func, jbtyp)
            print "Visualizing %16s %8s %8s (%s/%s)"%(compound, func, jbtyp, pos_num+1, tot_num),
            query_user = raw_input(': ')
            if query_user == 'y':
                with cd(folder):
                    pos = Structure(pos_filename)
                    pos.visualize()
                    pos_num += 1
            elif query_user == 'n':
                pos_num += 1
                # pass
            elif query_user == 'q':
                return
            elif query_user == '':
                pos_num += 1
                # pass
            elif query_user == 'l':
                # display legend
                for ss, struct in enumerate(structure_files):
                    print "%s/%s: %16s %8s %8s %8s"%(ss+1, tot_num, struct[0], struct[1], struct[2], struct[3])
            else:
                try:
                    num = int(float(query_user))
                    if 0 < num and num < tot_num:
                        pos_num = num-1
                    else: print 'Warning: Invalid index'
                except:
                    print 'Warning: Unrecognized input, try %s'%nav_options

    def gen_inputs(self, cluster, funcs=['pbe', 'scan'],
                   minkdens=700.0, ratiotol=0.2, maxkfactor=1.5,
                   encut=600.0, ismear_relax=1, ismear_static=-5,
                   sigma=0.2, sigma_molecule=0.05, ediffg=-0.0001):
        '''Generate input files for each compound

        cluster:                         cluster for the job scripts
        funcs:                           list of functional strings
        minkdens, ratiotol, maxkfactor:  parameters for kpoints
        encut:                           energy cutoff in eV
        ismear_relax:                    ismear for relaxation runs
        ismear_static:                   ismear for static runs
        sigma:                           smearing for relaxation runs
        ediffg:                          relaxation conv criterion

        Note: ismear is forced to be 0 for molecule static run

        '''
        # Generate the POSCARS if not already done
        if len(self.poscars) == 0: self.gen_poscars()
        print "[Generating input files]"

        # Save the basic settings for reference
        w = csv.writer(open(os.path.join(self.basedir, 'calc_settings.csv'), 'w'))
        for key, val in self.calc_settings.items():
            w.writerow([key, val])
            print "%15s = %20s"%(key, val)

        # Generate VASP inputs for each calculation
        for compound in self.systems:
            for func in self.calc_settings['funcs']:
                for jobtype in ['relax', 'rel_geom']:
                    folder = os.path.join(self.basedir, 'dft_runs', compound, func, jobtype)
                    if not os.path.isdir(folder):
                        os.makedirs(folder)
                        with cd(folder):
                            # keep initial poscar
                            pos = Structure(self.poscars[compound])
                            pos.title = compound
                            pos.pos2vasp('../POSCAR')
                            print "Creating %8s %8s %8s"%(compound, func, jobtype)
                            # try spin-polarized calculation if there is magnetic element
                            # or for triplet O2
                            try_magnetic = any([element in magnetic_els for element in Composition(compound).elements()])
                            ispin = 2 if try_magnetic or compound == 'O' else 1
                            if jobtype == 'relax':
                                pos.pos2vasp('POSCAR')
                                pos.gen_potcar()
                                dimen = 0 if compound in self.non_solid_elements else 3
                                km = Kmesh(pos)
                                km.gen_kmesh(minkdens=minkdens, dim=dimen, ratiotol=ratiotol, maxkfactor=maxkfactor, verbosity=1)
                                km.print_kmesh(out='KPOINTS')
                                # Gaussian smearing for molecule
                                ismear = 0 if compound in self.non_solid_elements else ismear_relax
                                # smaller smearing for molecule
                                sig = sigma_molecule if compound in self.non_solid_elements else sigma
                                # don't relax cell if diatomic molecule
                                isif = 2 if compound in self.non_solid_elements else 3
                                # want triplet state for O2
                                nupdown = 2 if compound == 'O' else 'None'
                                # initial parallel magnetic moments (ferromagnetic state) of 3.5 mu_B on each magnetic ions
                                if try_magnetic:
                                    magmom = ' '.join(['%i*%.2f'%(pos.natoms[atypind], 3.5 if pos.atomtypes[atypind] in magnetic_els else 0) for atypind in range(len(pos.natoms))])
                                elif compound == 'O':
                                    magmom = 'None'
                                else: magmom = ''
                                inc = Incar(pos)
                                inc.gen_incar(relax=True, func=func, ibrion=1, isif=isif, ediffg=ediffg, ismear=ismear_relax, sigma=sig, encut=encut, nelm=140, lcharg=True, ispin=ispin, magmom=magmom, nupdown=nupdown)
                                inc.set_nbands()
                                inc.print_incar('INCAR')
                                generate_jobscript(cluster, {'jobtype': 'vasp_scan', 'nnodes': 1, 'nhours': int(round(6.3*pos.totalnatoms*ispin))}, out='run.vasp')
                            elif jobtype == 'rel_geom':
                                execute_local("cp ../relax/POTCAR ../relax/KPOINTS .")
                                # Gaussian smearing for molecule
                                ismear = 0 if compound in self.non_solid_elements else ismear_static
                                # smaller smearing for molecule
                                sig = sigma_molecule if compound in self.non_solid_elements else sigma
                                # want triplet state for O2
                                nupdown = 2 if compound == 'O' else 'None'
                                inc = Incar(pos)
                                inc.gen_incar(func=func, ismear=ismear, sigma=sigma, encut=encut, nelm=200, lcharg=True, ispin=ispin)
                                inc.set_nbands()
                                if 'MAGMOM' in inc.incar: del inc.incar['MAGMOM'] # don't set
                                inc.print_incar('INCAR')
                                generate_jobscript(cluster, {'jobtype': 'vasp_scan', 'nnodes': 1, 'nhours': int(round(2.2*pos.totalnatoms*ispin))}, out='run.vasp')

    def submit_jobs(self, cluster, funcs='', debug=False):
        '''Submit jobs that are ready'''
        funcs = self._set_funcs(funcs)
        if cluster in down_clusters: sys.exit('Error: %s cluster is down'%cluster)
        # determine how many jobs in cluster queue
        if cluster == 'cypress' and funcs:
            showq_cmd = "showq -u mko570 | grep 'Total job'"
            out, err = execute_remote(cluster, showq_cmd)
            nslots = 500 - int(float(out.split()[-1]))
        elif cluster == 'edison':
            showq_cmd = '/usr/bin/squeue -u mkothako | grep -v JOBID | wc -l'
            out, err = execute_remote(cluster, showq_cmd)
            nslots = 99 - int(float(out))
        job_dirs = []
        for compound in self.systems:
            for func in funcs:
                for jobtype in ['relax', 'rel_geom']:
                    job_dir = os.path.join(self.basedir, 'dft_runs', compound, func, jobtype)
                    job_dirs.append(job_dir)
        statuses = self.job_statuses(folders=job_dirs, only_local=True)
        print "[%sSubmitting jobs that are ready on %s]"%('(Debug) ' if debug else '', cluster)
        for compound in self.systems:
            for func in funcs:
                for jobtype in ['relax', 'rel_geom']:
                    job_dir = os.path.join(self.basedir, 'dft_runs', compound, func, jobtype)
                    if job_dir in statuses:
                        status = statuses[job_dir]
                        if status[0] and not status[1]:
                            if nslots > 0:
                                with cd(job_dir):
                                    # ensure job script is in format for cluster
                                    js = Jobscript(cluster)
                                    js.read_jobscript(infile='run.vasp')
                                    nnodes = min(2, js.nnodes) ; nhours = min(36, js.nhours)
                                    generate_jobscript(cluster, {'jobtype': 'vasp_scan', 'nnodes': nnodes, 'nhours': nhours}, out='run.vasp')
                                    print "Submitting %16s %8s %8s"%(compound, func, jobtype)
                                    if not debug:
                                        execute_local("sleep 10 ; rm -f job.out ; ~/scripts/sendjob.sh %s %sscratch 2>&1 > job.out"%(cluster, 'my' if cluster == 'cypress' else ''))
                                        nslots -= 1
                                        # make sure it actually submitted
                                        try:
                                            jobid_txt = execute_local("grep -B 1 'Job submitted' job.out | head -1")[0]
                                            # cypress: just the number; edison: Submitted batch job 4489315
                                            if 'Moab' in jobid_txt:
                                                print 'Warning: %s job number contains Moab, suggesting problem with submission'%cluster
                                            else:
                                                jobid = int(float(jobid_txt)) if len(jobid_txt.split()) == 1 else int(float(jobid_txt.split()[3]))
                                        except Exception as e:
                                            sys.exit('Error: Problem submitting %s due to error %s'%(job_dir, e))
                            else: sys.exit("Error: %s ready to submit, but %s queue is too full"%(job_dir, cluster))
                    else: sys.exit('Error: Job status for %s not found'%job_dir)

    def cancel_jobs(self, clusters, include_running_jobs=False, funcs=''):
        '''Cancel queued or all jobs'''
        funcs = self._set_funcs(funcs)
        job_dirs = []
        for compound in self.systems:
            for func in funcs:
                for jobtype in ['relax', 'rel_geom']:
                    job_dir = os.path.join(self.basedir, 'dft_runs', compound, func, jobtype)
                    job_dirs.append(job_dir)
        statuses = self.job_statuses(job_dirs, clusters)
        jobs_to_cancel = []
        for job_dir in statuses:
            ready, submitted, queued, running, cluster, jobid, processed = statuses[job_dir]
            satisfied = [ready, submitted, queued, cluster in clusters, not processed]
            if not include_running_jobs: satisfied.append(not running)
            if all(satisfied):
                jobs_to_cancel.append([jobid, cluster, job_dir])
        print "[Canceling %sjobs on %s]"%('all ' if include_running_jobs else 'not-yet-running ', ','.join(clusters))
        if len(jobs_to_cancel) > 0:
            print '\n'.join([' '.join([str(jj) for jj in job]) for job in jobs_to_cancel])
            self._cancel_jobs(jobs_to_cancel)

    def _cancel_jobs(self, jobs_to_cancel):
        '''Helper function to cancel a bunch of jobs together'''
        clusters = set([job[1] for job in jobs_to_cancel]) # unique ones
        for cluster in clusters:
            jobids = [job[0] for job in jobs_to_cancel if job[1] == cluster]
            folders = [job[2] for job in jobs_to_cancel if job[1] == cluster]
            remote_command = "%s %s && rm -r ~/%sscratch{%s}"%('qdel' if cluster == 'cypress' else 'scancel', ' '.join([str(jid) for jid in jobids]), 'my' if cluster == 'cypress' else '', ','.join(folders))
            execute_remote(cluster, remote_command)
            local_command = "rm -f {%s}/job.out"%(','.join(folders))
            execute_local(local_command)

    def job_statuses(self, folders, clusters='', only_local=False, verbosity=0):
        '''Check job statuses

        Returns a dict, which for each key (folder) contains another
        list with several elements:
        0   logical  job_ready       is job ready to run?
        1   logical  job_submitted   has job been submitted?
        2   logical  job_in_queue    if submitted, is job in the queue?
        3   logical  job_running     if submitted, is job currently running?
        4   string   cluster         if submitted, which cluster? (else 'None')
        5   integer  jobid           if submitted, the jobid (else -1)
        6   logical  job_processed   has been processed?

        only_local:  don't do anything requiring remote server ssh

        '''
        if clusters == '': clusters = known_clusters
        if len(folders) > 1: print "[Querying %sjob statuses for %s]"%('local ' if only_local else '', ','.join(clusters))
        job_statuses = {folder: ['' for index in range(7)] for folder in folders} # initialize
        to_check_remote = []
        for folder in folders:
            job_statuses[folder][4] = 'None' # cluster
            job_statuses[folder][5] = -1 # jobid
            with cd(folder):
                # ready if non-empty VASP files
                infiles = ['INCAR', 'POSCAR', 'POTCAR', 'KPOINTS']
                job_statuses[folder][0] = all([os.path.isfile(infile) and os.path.getsize(infile) != 0 for infile in infiles]) # job_ready
                # check if job has been submitted/processed via job.out file
                job_statuses[folder][1] = os.path.isfile('job.out') # job_submitted
                if not job_statuses[folder][1]:
                    job_statuses[folder][3] = False # job_running
                    job_statuses[folder][6] = False # job_processed
                else:
                    if int(float(execute_local("grep 'Local dir' job.out | wc -l")[0])) == 0:
                        # no Local dir in job.out
                        if os.path.exists('OSZICAR'):
                            # found OSZICAR, so must be done and processed and cluster overwrote job.out
                            job_statuses[folder][3] = False # job running
                            job_statuses[folder][6] = True # job processed
                        else:
                            # no OSZICAR, so something wrong
                            sys.exit('Error: Job in %s directory appears to be processed, but cannot find OSZICAR'%folder)
                    else:
                        # found Local dir in job.out
                        if os.path.exists('OSZICAR'):
                            # also found OSZICAR, so must be done and processed
                            job_statuses[folder][3] = False # job running
                            job_statuses[folder][6] = True # job processed
                        else:
                            # no OSZICAR, so must be not yet processed
                            job_statuses[folder][6] = False # job processed
                if job_statuses[folder][6] and not job_statuses[folder][0]: sys.exit('Error: Job in %s directory is missing input files, but it has already been processed?')
                if job_statuses[folder][1] and not job_statuses[folder][0]: sys.exit('Error: Job in %s directory is missing input files, but it has already been submitted?')
                if job_statuses[folder][0] and job_statuses[folder][1] and not job_statuses[folder][6]:
                    # determine cluster from job.out file
                    try:
                        job_statuses[folder][4] = execute_local("grep 'Remote dir' job.out | head -1")[0].split(':')[1].strip()
                    except:
                        sys.exit('Error: Cannot determine cluster from job.out file in %s directory'%folder)
                    if job_statuses[folder][4] in known_clusters:
                        if job_statuses[folder][4] in clusters:
                            # determine job id from job.out file
                            try:
                                jobid_txt = execute_local("grep -B 1 'Job submitted' job.out | head -1")[0].replace('Moab', '')
                                # cypress: just the number; edison: Submitted batch job 4489315
                                job_statuses[folder][5] = int(float(jobid_txt)) if len(jobid_txt.split()) == 1 else int(float(jobid_txt.split()[3]))
                            except:
                                # print 'Could not find job ID in job.out. Removing job.out and directory on %s.'%job_statuses[folder][4]
                                # with cd(folder):
                                #     execute_local('rm -f job.out')
                                #     execute_local('ssh %s "rm -rf ~/%sscratch/%s"'%(job_statuses[folder][4], 'my' if job_statuses[folder][4] == 'cypress' else '', folder))
                                with cd(folder):
                                    out, err = execute_local('pwd ; ls job.out ; cat job.out')
                                sys.exit('Error: Could not find job ID in job.out in %s directory'%folder)
                            to_check_remote.append(folder)
                    else:
                        print 'cluster', job_statuses[folder][4]
                        print 'known_clusters', known_clusters
                        print 'down_clusters', down_clusters
                        sys.exit('Error: %s cluster is not implemented'%job_statuses[folder][4])
        # parts in which we need ssh
        if not only_local:
            for cluster in [clst for clst in clusters if clst not in down_clusters]:
                jobids = [job_statuses[job_dir][5] for job_dir in to_check_remote if job_statuses[job_dir][4] == cluster]
                folders = [job_dir for job_dir in to_check_remote if job_statuses[job_dir][4] == cluster]
                if len(jobids) > 0:
                    # check if in queue or running (multiple lines for qstat/squeue and not complete status)
                    qstat_remote_command = "for jobid in %s ; do line=$(%s $jobid 2>/dev/null | grep -v 'Name' | grep -v 'USER' | grep -v '\-\-') ; if [ '$line' == '' ] ; then echo ; else echo $line ; fi ; done"%(' '.join([str(jid) for jid in jobids]), 'qstat' if cluster == 'cypress' else 'squeue -j')
                    qstat_result, qstat_err = execute_remote(cluster, qstat_remote_command)
                    # check for vasp.out presence
                    ls_remote_command = "for dir in %s ; do ls ~/%sscratch${dir}/vasp.out 2>/dev/null | wc -l ; done"%(' '.join(folders), 'my' if cluster == 'cypress' else '')
                    exists_on_cluster, err_exists_on_cluster = execute_remote(cluster, ls_remote_command, error_ok=True, print_warning=True)
                    if len(qstat_result.splitlines()) != len(exists_on_cluster.splitlines()):
                        print
                        print 'qstat_remote_command'
                        print qstat_remote_command
                        print 'ls_remote_command'
                        print ls_remote_command
                        print 'qstat_result', len(qstat_result.splitlines())
                        print qstat_result
                        print 'exists_on_cluster', len(exists_on_cluster.splitlines())
                        print exists_on_cluster
                        sys.exit('Error: Inconsistency in output of %s and ls commands checking jobs on %s'%('qstat' if cluster == 'cypress' else 'squeue -j', cluster))
                    if len(qstat_result.splitlines()) != len(folders):
                        print
                        print 'qstat_remote_command', qstat_remote_command
                        print 'qstat_result', qstat_result
                        print 'folders', folders
                        print 'jobids', jobids
                        print 'without redirecting stderr to dev null:'
                        print qstat_remote_command.replace('2>/dev/null ', '')
                        out, err = execute_remote(cluster, qstat_remote_command.replace('2>/dev/null ', ''))
                        print out
                        print err
                        sys.exit('Error: %s did not give the expected number of lines on %s'%('qstat' if cluster == 'cypress' else 'squeue -j', cluster))
                    for it, folder in enumerate(folders):
                        try:
                            int(float(exists_on_cluster.splitlines()[it]))
                        except:
                            sys.exit("Error: Could not process result of ls ~/%sscratch%s/vasp.out | wc -l on %s"%('my' if job_statuses[folder][4] == 'cypress' else '', folder, cluster))
                        status_column = 4 if cluster == 'edison' else -2
                        if len(qstat_result.splitlines()[it].split()) > 1 and qstat_result.splitlines()[it].split()[status_column] not in ['C', 'CD']:
                            # in queue (queued or running)
                            job_statuses[folder][2] = True
                            if int(float(exists_on_cluster.splitlines()[it])) == 1:
                                # vasp.out present, so must be running
                                # cannot be finished since would not be in queue if so
                                job_statuses[folder][3] = True
                            else:
                                job_statuses[folder][3] = False
                        elif int(float(exists_on_cluster.splitlines()[it])) == 1:
                            # vasp.out present but not in queue, so must be finished
                            job_statuses[folder][2] = False
                            job_statuses[folder][3] = False
                        else:
                            sys.exit('Error: Cannot determine job status for %s job on %s'%(folder, cluster))
                            # print 'Warning: Cannot determine job status for %s job on %s. Removing job.out.'%(folder, cluster)
                            # job_statuses[folder][1] = False
                            # job_statuses[folder][2] = False
                            # job_statuses[folder][3] = False
                            # with cd(folder):
                            #     execute_local('rm -f job.out')
        # return the dict
        return job_statuses

    def check_relax(self, cluster, funcs=''):
        '''Grab and examine finished relaxation runs

        funcs: list of functional strings

        '''
        funcs = self._set_funcs(funcs)
        job_dirs = []
        for compound in self.systems:
            for func in funcs:
                job_dir = os.path.join(self.basedir, 'dft_runs', compound, func, 'relax')
                job_dirs.append(job_dir)
        statuses = self.job_statuses(folders=job_dirs)
        print "[Checking on relaxation runs]"
        for compound in self.systems:
            for func in funcs:
                print "%12s %8s   "%(compound, func),
                folder = os.path.join(self.basedir, 'dft_runs', compound, func, 'relax')
                job_complete = check_job(folder, statuses)
                if job_complete:
                    with cd(folder):
                        # check if actually done or needs to be resubmitted
                        converged = 'reached required accuracy' in execute_local("tail -1 vasp.out 2>/dev/null")[0]
                        if not converged:
                            fix_vasp_errors(folder)
                            prev_incar = Incar(Structure('POSCAR'))
                            prev_incar.read_incar('INCAR')
                            if os.path.isfile('CONTCAR') and os.path.getsize('CONTCAR') > 0:
                                print "Not converged. Setting up again with final structure. Adding walltime and/or nodes."
                                # if spin-polarized, re-initialize with most recent moments
                                ispin = int(float(execute_local('grep ISPIN OUTCAR')[0].split()[2]))
                                if ispin == 2:
                                    magmoms = execute_local('/home/mkothako/scripts/get_moms.sh')[0]
                                    prev_incar.incar['MAGMOM'] = magmoms
                                # swap between IBRION of 1 and 2
                                if prev_incar.incar['IBRION'] in [1, 2]:
                                    prev_incar.incar['IBRION'] = 3 - prev_incar.incar['IBRION']
                                execute_local("mv CONTCAR POSCAR")
                                cleanup_vasp(folder, remove_extras=True, archive=True, keep=['CHGCAR'])
                                new_res = add_resources(cluster, folder=folder)
                                if not new_res: print "Warning: Cannot increase computational resources"
                            else:
                                # no CONTCAR so didn't make it even 1 ionic step --> add resources
                                print "Not converged. Adding walltime and/or nodes."
                                cleanup_vasp(folder, remove_extras=True, archive=True, keep=['CHGCAR'])
                                new_res = add_resources(cluster, folder=folder)
                                if not new_res: print "Warning: Cannot increase computational resources"
                            prev_incar.print_incar('INCAR')
                        else:
                            print "Relaxation has converged. Copying structure to static run."
                            if os.path.isdir('../rel_geom'):
                                execute_local('cp CONTCAR ../rel_geom/POSCAR')
                                # if spin-polarized, copy final relaxation moments to static initialization
                                ispin = int(float(execute_local('grep ISPIN OUTCAR')[0].split()[2]))
                                if ispin == 2:
                                    magmoms = execute_local('/home/mkothako/scripts/get_moms.sh')[0]
                                    INC = open('../rel_geom/INCAR', 'a') ; INC.write("MAGMOM     = %s"%magmoms) ; INC.close()
                            else: sys.exit("Error: ../rel_geom directory does not exist.")

    def check_static(self, cluster, funcs=''):
        '''Grab and examine finished static runs

        funcs: list of functional strings

        '''
        funcs = self._set_funcs(funcs)
        job_dirs = []
        for compound in self.systems:
            for func in funcs:
                job_dir = os.path.join(self.basedir, 'dft_runs', compound, func, 'rel_geom')
                job_dirs.append(job_dir)
        statuses = self.job_statuses(folders=job_dirs)
        print "[Checking on static runs]"
        for compound in self.systems:
            for func in funcs:
                print "%12s %8s   "%(compound, func),
                folder = os.path.join(self.basedir, 'dft_runs', compound, func, 'rel_geom')
                job_complete = check_job(folder, statuses)
                if job_complete:
                    with cd(folder):
                        # check if converged or needs to be resubmitted
                        try:
                            nelm = int(float(execute_local("grep NELM OUTCAR | awk '{print $3}'")[0].replace(';', '')))
                        except:
                            nelm = -1 # must not be converged if can't find NELM in OUTCAR
                        last_two = execute_local("tail -n 2 vasp.out")[0].splitlines()
                        try:
                            converged = 'E0=' in last_two[1] and last_two[0] != '' and int(float(last_two[0].split()[1])) < nelm
                        except:
                            converged = False
                            print 'Warning: Potential unknown error in vasp.out in %s'%folder
                            print 'last_two', last_two
                        if not converged:
                            fix_vasp_errors(folder)
                            print "Not converged. Adding walltime and/or nodes."
                            cleanup_vasp(folder, remove_extras=True, archive=True, keep=['CHGCAR'])
                            new_res = add_resources(cluster, folder=folder)
                            if not new_res: print "Warning: Cannot increase computational resources"
                        else: print "Static run has converged."

    def _set_funcs(self, infuncs):
        '''Helper function to set funcs

        infuncs: list of functional strings

        '''
        # If funcs not set, try to get from self.calc_settings
        if infuncs == '':
            if 'funcs' in self.calc_settings:
                outfuncs = self.calc_settings['funcs']
            else: sys.exit('Error: Must specify funcs if not set in calc_settings dict')
        else: outfuncs = infuncs
        return outfuncs

    def read_calc_settings(self, filename='calc_settings.csv'):
        '''Read in calculation settings from csv file'''
        print "[Reading in calculation settings from %s]"%filename
        with open(os.path.join(self.basedir, filename), 'r') as calcsettings:
            reader = csv.reader(calcsettings)
            for row in reader:
                if row[0] == 'funcs':
                    self.calc_settings[row[0]] = list(row[1].replace("'", '').replace('[', '').replace(']', '').replace(' ', '').split(',')) # list
                elif 'ismear' in row[0]:
                    self.calc_settings[row[0]] = int(float(row[1])) # int
                else:
                    self.calc_settings[row[0]] = float(row[1]) # float

    def read_gaps(self, filename='exp_bandgaps.csv'):
        '''Read in experimental band gaps from csv'''
        print "[Reading in experimental band gaps from %s]"%filename
        with open(os.path.join(self.basedir, filename), 'rU') as bandgaps:
            reader = csv.reader(bandgaps)
            for row in reader:
                if row[0] == 'Material' or row[0] == 'Compound':
                    pass # header
                elif row[0][0] == '#':
                    pass # comment
                elif is_number(row[1]):
                    self.exp_gaps[Composition(row[0]).format_string()] = float(row[1])
                else:
                    sys.exit('Error: Cannot understand band gap value of %s for %s in %s'%(row[1], row[1], filename))

    def kpt_table(self, jobtype='rel_geom', out='kpoints.txt'):
        '''Generate a table with k-point information

        Each row contains the compound, the k-mesh, the k-pt density,
        and the KPPRA

        '''
        OUT = open(os.path.join(self.basedir, out), 'w')
        print "[Generating k-point table]"
        header = ['%18s'%'Compound', '%8s'%'k-mesh', '%7s'%'kpt-dens', '%7s'%'kppra']
        print '  '.join(header) ; OUT.write('  '.join(header) + '\n')
        kpt_rows = []
        for compound in self.systems:
            if compound not in gas:
                # just looking at pbe directories
                folder = os.path.join(self.basedir, 'dft_runs', compound, 'pbe', jobtype)
                with cd(folder):
                    if os.path.exists('POSCAR'):
                        pos = Structure('POSCAR')
                    elif os.path.exists('../relax/POSCAR'):
                        pos = Structure('../relax/POSCAR')
                    else:
                        # sys.exit('Error: Cannot find POSCAR')
                        pass
                    km = Kmesh(pos)
                    km.read_kmesh()
                    kdens = km.print_kdens(ret=True)
                    kppra = km.print_kppra(ret=True)
                    row = ['%18s'%compound, '%2i %2i %2i'%tuple(km.kmesh), '%7.2f'%kdens, '%7s'%kppra]
                    kpt_rows.append(row)
                    print '  '.join(row) ; OUT.write('  '.join(row) + '\n')
        OUT.close()
        # stats on k-points
        kdens_array = np.array([float(rw[2]) for rw in kpt_rows])
        kppra_array = np.array([float(rw[3]) for rw in kpt_rows])
        print 'Mean Kdens: %16.3f'%(np.mean(kdens_array))
        print ' Min Kdens: %16.3f'%(np.min(kdens_array))
        print ' Max Kdens: %16.3f'%(np.max(kdens_array))
        print ' Std Kdens: %16.3f'%(np.std(kdens_array))

        print 'Mean KPPRA: %16i'%(np.mean(kppra_array))
        print ' Min KPPRA: %16i'%(np.min(kppra_array))
        print ' Max KPPRA: %16i'%(np.max(kppra_array))
        print ' Std KPPRA: %16i'%(np.std(kppra_array))

    def check_potcars(self):
        '''Check that the pseudopotentials used for the element calculations
        have the kinetic energy density'''
        for system in self.systems:
            if Composition(system).nelements() == 1:
                # just looking at pbe directories
                folder = os.path.join(self.basedir, 'dft_runs', system, 'pbe', 'rel_geom')
                with cd(folder):
                    print system
                    print execute_local('grep kinetic POTCAR')[0]
                    print

    def gather_fe(self, funcs='', sources=['iit', 'ssub'],
                  include_zpe=False, ignore_oqmd_dftu=False, fe_file='fe.yml',
                  shift_mus=False, only_exp=False, from_scratch=False):
        '''Build the formation energy dictionary

        include_zpe: Include phonon ZPE corrections for diatomics
        (must be in self.zpe dict as generated by self.gather_zpe())

        ignore_oqmd_dftu: Ignore compounds for which OQMD uses DFT+U,
        to facilitate proper comparison

        fe_file: separate YAML file containing formation energies
        (used to avoid regrabbing experimental values from larue)

        shift_mus: if True, read chemical potential shifts from
        mu_shifts.yaml

        '''
        # Generate the POSCARS if not already done
        if len(self.poscars) == 0: self.gen_poscars()
        funcs = self._set_funcs(funcs)
        if from_scratch: self.fe = {}
        # dictionary of exp fe values to fix
        fix_exp_fe = {'LiNbO3': -2.774, 'CaAl2': -0.346, 'ZrPt':
                      -0.99, 'AlB2': -0.055, 'InN': -0.148}
        # read in exp fe already grabbed
        exp_fe_data = yaml.load(open(fe_file, 'r')) if os.path.isfile(fe_file) else {}
        print "[Gathering experimental (%s) %sformation energies]"%(', '.join(sources), 'and calculated (%s) '%(', '.join(funcs)) if not only_exp else '')
        for compound in sort_systems(self.compounds):
            if ignore_oqmd_dftu and oqmd_dftu(compound):
                pass
            else:
                self.fe[compound] = {}
                # grab experimental value(s)
                if compound in exp_fe_data and 'exp' in exp_fe_data[compound]:
                    # using already grabbed data
                    for source in sources:
                        if source in exp_fe_data[compound]['exp']:
                            if 'exp' not in self.fe[compound]: self.fe[compound]['exp'] = {}
                            self.fe[compound]['exp'][source] = exp_fe_data[compound]['exp'][source]
                else:
                    # grabbing data from larue
                    for source in sources:
                        exists, exp_fes = self.grab_exp_fe(compound, source)
                        if exists:
                            if 'exp' not in self.fe[compound]: self.fe[compound]['exp'] = {}
                            self.fe[compound]['exp'][source] = exp_fes
                if compound in fix_exp_fe and compound in self.fe and 'exp' in self.fe[compound] and 'ssub' in self.fe[compound]['exp']:
                    if self.fe[compound]['exp']['ssub'] != [fix_exp_fe[compound]]:
                        # if not already fixed, fix exp fe for the
                        # compounds in Table 5 of 2015 OQMD paper for
                        # which deviation magnitude b/w SSUB and
                        # alternative value is more than 0.1 eV/atom
                        print 'Note: Replacing exp. f.e. for %s with alternative value (see Table 5 of 2015 OQMD paper and discussion)'%(compound)
                        self.fe[compound]['exp'] = {} # remove any iit values
                        self.fe[compound]['exp']['ssub'] = [fix_exp_fe[compound]]
                if len(self.fe[compound]['exp']) == 0: print 'Warning: No experimental f.e. found for %s'%compound
                # check if large discrepancy b/w diff exp values
                exp_values = []
                if 'ssub' in self.fe[compound]['exp']: exp_values.extend(self.fe[compound]['exp']['ssub'])
                if 'iit' in self.fe[compound]['exp']: exp_values.extend(self.fe[compound]['exp']['iit'])
                if any([any([abs(100*(exp_val_2-exp_val_1)/exp_val_1) > 100. for exp_val_2 in exp_values]) for exp_val_1 in exp_values]):
                    print 'Warning: Largest variation between a pair of exp f.e. values (%7.1f%%) is greater than 100%% for %s'%(max([max([abs(100*(exp_val_2-exp_val_1)/exp_val_1) for exp_val_2 in exp_values]) for exp_val_1 in exp_values]), compound)
                if not only_exp:
                    # grab calculated value(s)
                    for func in funcs:
                        exists, calc_fe = self.compute_calc_fe(compound, func, include_zpe, shift_mus=shift_mus)
                        if exists:
                            if 'calc' not in self.fe[compound]: self.fe[compound]['calc'] = {}
                            self.fe[compound]['calc'][func] = calc_fe
        # save exp fe to enable faster gather in future
        fe_copy = self.fe.copy()
        for compound in exp_fe_data:
            # include any not used in current run (e.g. if filtering)
            if compound not in fe_copy:
                fe_copy[compound] = exp_fe_data[compound]
        with open(fe_file, 'w') as outfile:
            yaml.dump(fe_copy, outfile, default_flow_style=False)

    def grab_exp_fe(self, compound, source):
        '''Grab the experimental formation energy value(s)

        compound:  compound string
        source:    source

        '''
        grab_fe = execute_remote('larue', "/home/mko570/scripts/entry2fe.sh %i %s"%(self.compounds[compound], source))[0]
        fes = [float(fe) for fe in grab_fe.split()]
        if len(fes) > 0:
            return True, list(set(fes)) # avoid duplicates
        else: return False, [99999.9]

    def compute_calc_fe(self, compound, func, include_zpe=False, shift_mus=False):
        '''Compute the calculated formation energy value

        compound: compound string

        func: functional

        include_zpe: include ZPE correction?

        shift_mus: if True, read chemical potential shifts from
        mu_shifts.yaml

        '''

        # Read any chemical potential shifts from mu_shifts.yaml if
        # have not already
        if shift_mus:
            if not hasattr(self, 'mu_shifts'):
                if os.path.isfile('mu_shifts.yaml'):
                    print '[Reading mu shifts from mu_shifts.yaml]'
                    self.mu_shifts = yaml.load(open('mu_shifts.yaml', 'r'))
                else:
                    sys.exit('Error: Chemical potential shifting recypressed, but mu_shifts.yaml does not exist')
        else: self.mu_shifts = {}
        compound_folder = os.path.join(self.basedir, 'dft_runs', compound, func, 'rel_geom')
        check_ecompound = self.total_energy(compound_folder)
        if check_ecompound == False:
            return False, 99999.9 # insufficient info
        else:
            ecompound = check_ecompound[0] # total
            pos = Structure(self.poscars[compound].split('\n'))
            fe = ecompound # first term
            for el in pos.atomtypes:
                # subtract off contribution from each elemental reference
                el_folder = os.path.join(self.basedir, 'dft_runs', el, func, 'rel_geom')
                check_eatom = self.total_energy(el_folder)
                if check_eatom == False:
                    return False, 99999.9 # insufficient info
                else:
                    eatom = check_eatom[1] # per atom
                    if include_zpe and el in gas and el != 'Xe':
                        try:
                            eatom += self.zpe[el]['calc'][func]
                        except:
                            sys.exit('Error: Cannot find ZPE correction for %s in %s with %s functional'%(el, compound, func))
                    if el in self.mu_shifts and func in self.mu_shifts[el]:
                        eatom += self.mu_shifts[el][func]
                    fe = fe - (eatom * pos.natoms[pos.atomtypes.index(el)])
            fe = fe / pos.totalnatoms
            return True, fe

    def total_energy(self, folder):
        '''Compute the energy (total and per atom)'''
        if not os.path.isdir(folder):
            print 'Warning: %s directory does not exist'%folder
            return False
        with cd(folder):
            if not all([os.path.isfile(needed_file) for needed_file in ['POSCAR', 'OSZICAR']]):
                print 'Warning: OSZICAR and/or POSCAR does not exist in %s'%folder
                return False
            else:
                # grab the total energy
                lasph = len(execute_local("grep ASPHER OSZICAR")[0].splitlines())
                if lasph > 1: sys.exit("Error: Found more than one line with ASPHER in OSZICAR file.")
                if lasph == 1:
                    etot = float(execute_local("grep ASPHER OSZICAR | awk 'END {print $5}'")[0])
                else:
                    etot = float(execute_local("grep E0= OSZICAR | awk 'END {print $5}'")[0])
                # determine number of atoms in the cell
                pos = Structure('POSCAR')
                natoms = pos.totalnatoms
                return etot, etot/pos.totalnatoms

    def gather_zpe(self, funcs=''):
        '''Gather the computed and experimental ZPE for the diatomics'''
        funcs = self._set_funcs(funcs)
        # experimental values in eV/atom
        irikura = {'H': 136.4/1000., 'N': 73.1/1000., 'O': 49.0/1000., 'F': 28.4/1000., 'Cl':17.3/1000.}
        for el in self.non_solid_elements:
            if el != 'Xe':
                self.zpe[el] = {}
                # experimental values from Irikura, J. Phys. Chem. Ref. Data 36, 389 (2007)
                self.zpe[el]['exp'] = {}
                self.zpe[el]['exp']['irikura'] = irikura[el]
                # calculated values from finite difference calculation,
                # which should be manually run in folder structure like
                # 'H/pbe/rel_geom/phonons'
                for func in funcs:
                    folder = os.path.join(self.basedir, 'dft_runs', el, func, 'rel_geom', 'phonons')
                    with cd(folder):
                        phonon_energy = float(execute_local("grep -A 4 'dynamical' OUTCAR | tail -n 1")[0].split()[-2])/1000.
                        if 'calc' not in self.zpe[el]: self.zpe[el]['calc'] = {}
                        # factor of 2 from ZPE equation
                        # factor of 2 since per atom and 2 atoms in diatomic molecule
                        self.zpe[el]['calc'][func] = phonon_energy/4.

    def print_fe(self, funcs='', sources=['iit', 'ssub'], avg_expt=False):
        '''Print the formation energy dictionary'''
        funcs = self._set_funcs(funcs)
        if len(self.fe) > 0:
            print '-'*90
            print 'compound'.center(15),
            func_header = '' ; source_header = ''
            for func in funcs: func_header += '%10s'%func.center(10)
            print func_header,
            for source in sources: source_header += '%25s'%source.center(25)
            print source_header
            print '-'*90
            for comp in sort_systems(self.fe):
                fe_string = '%14s'%(Composition(comp).format_string('latex')).center(15)
                for func in funcs:
                    if 'calc' in self.fe[comp] and func in self.fe[comp]['calc']:
                        fe_string += ('%10.3f'%self.fe[comp]['calc'][func]).center(10)
                    else:
                        fe_string += '%10s'%' '.center(10)
                for source in sources:
                    if 'exp' in self.fe[comp] and source in self.fe[comp]['exp']:
                        fe_string += '%25s'%('%.3f'%np.mean(self.fe[comp]['exp'][source]) if avg_expt else '/'.join(['%.3f'%val for val in self.fe[comp]['exp'][source]])).center(25)
                    else:
                        fe_string += '%25s'%' '.center(25)
                print fe_string
            print '-'*90
            print '%i compounds'%len(self.fe)

    def save_fe(self, funcs='', sources=['iit', 'ssub'], avg_expt=False, outfile='fe.csv'):
        '''Save the formation energy dictionary'''
        funcs = self._set_funcs(funcs)
        print "[Saving formation energies as %s]"%outfile
        if len(self.fe) > 0 or os.path.isfile(outfile):
            fe_table = []
            for comp in sort_systems(self.fe):
                fe_table_entry = []
                fe_table_entry.append(comp)
                for func in funcs:
                    if 'calc' in self.fe[comp] and func in self.fe[comp]['calc']:
                        fe_table_entry.append(self.fe[comp]['calc'][func])
                    else: fe_table_entry.append(None)
                for source in sources:
                    if 'exp' in self.fe[comp] and source in self.fe[comp]['exp']:
                        fe_table_entry.append(np.mean(self.fe[comp]['exp'][source]) if avg_expt else '/'.join(['%s'%val for val in self.fe[comp]['exp'][source]]))
                    else: fe_table_entry.append(None)
                fe_table.append(fe_table_entry)
            w = csv.writer(open(os.path.join(self.basedir, outfile), 'w'))
            header_list = []
            header_list.append('Compound')
            for func in funcs: header_list.append('calc_'+func)
            for source in sources: header_list.append('exp_'+source)
            w.writerows([header_list])
            for fe_row in fe_table: w.writerow(fe_row)

    def read_fe(self, infile='fe.csv', append=False, overwrite=True):
        '''Read in the formation energies from a file'''
        funcs = [] ; sources = []
        print "[Reading in formation energies from %s]"%infile
        # Determine the functionals and exp sources from header
        # Note: need calc_ or exp_ prefixes and all calc must be before all exp_
        first_line = open(os.path.join(self.basedir, infile), 'r').readlines()[0]
        if 'Compound' in first_line:
            for func_source in first_line.split(',')[1:]:
                if 'calc' in func_source:
                    funcs.append(func_source.replace('calc_', '').rstrip())
                elif 'exp' in func_source:
                    sources.append(func_source.replace('exp_', '').rstrip())
                else: sys.exit('Error: Header f.e. entries should be preceeded by calc_ or exp_')
        else: sys.exit('Error: Incorrect header format for %s'%infile)
        if not append: self.fe = {} # ignore old info
        reader = csv.reader(open(os.path.join(self.basedir, infile), 'r'))
        for row in reader:
            if 'Compound' not in row:
                for ff, func in enumerate(funcs):
                    fe_val = row[1+ff]
                    if fe_val:
                        if row[0] not in self.fe: self.fe[row[0]] = {}
                        if 'calc' not in self.fe[row[0]]: self.fe[row[0]]['calc'] = {}
                        if func not in self.fe[row[0]]['calc']:
                            self.fe[row[0]]['calc'][func] = float(fe_val)
                        else:
                            if overwrite:
                                print 'Warning: Overwriting f.e. for %s - %s'%(row[0], func)
                                self.fe[row[0]]['calc'][func] = float(fe_val)
                            else: print 'Warning: Leaving %s - %s value unchanged'%(row[0], func)
                for ss, source in enumerate(sources):
                    fe_val = row[1+len(funcs)+ss]
                    if fe_val:
                        if row[0] not in self.fe: self.fe[row[0]] = {}
                        if 'exp' not in self.fe[row[0]]: self.fe[row[0]]['exp'] = {}
                        if source not in self.fe[row[0]]['exp']:
                            self.fe[row[0]]['exp'][source] = [float(vv) for vv in fe_val.split('/')]
                        else:
                            if overwrite:
                                print 'Warning: Overwriting f.e. for %s - %s'%(row[0], source)
                                self.fe[row[0]]['exp'][source] = [float(vv) for vv in fe_val.split('/')]
                            else: print 'Warning: Leaving %s - %s value unchanged'%(row[0], func)

    def optimize_single_anion_mu(self, funcs=''):
        '''Calculate the chemical potential correction for a single anion
        element that minimizes the formation energy MAE for a group of
        compounds containing the anion and no other anion elements'''

        funcs = self._set_funcs(funcs)
        # determine anion element
        anion_els = [anion_el for anion_el in ['H', 'N', 'O', 'F', 'Cl', 'Xe'] if Composition(self.fe.keys()[0]).contains(anion_el)]
        if len(anion_els) != 1:
            sys.exit('Error: Printing F.E. per anion element recypressed, but found != 1 diatomic molecule anion elements in %s'%self.fe.keys()[0])
        else:
            anion_el = anion_els[0]

        for func in funcs:
            opt_result = optimize.minimize_scalar(self.fe_mae, args=(anion_el, func))
            if opt_result['success']:
                print 'Best chemical potential shift for %2s using %4s functional is %.8f'%(anion_el, func.upper(), opt_result['x'])
            else:
                print 'Error: Optimization failed for %s functional. Error message:'%func.upper()
                print opt_result['message']
                sys.exit()

    def chemical_potential_fitting(self, funcs='', fit_els=[], quiet=False):
        '''Calculate the chemical potential corrections simultaneously for a
        set of elements

        We perform a least-squares fitting so the RMSE is minimized

        fit_els: list of element symbol strings (e.g. ['Fe', 'O'])

        Not a problem if fit_els contains an element absent in the
        entire compound set

        '''
        print "[Performing elemental chemical potential fitting]"
        funcs = self._set_funcs(funcs)

        # enable auto selection of all elements
        all_fit_els = fit_els == 'All'
        if all_fit_els:
            fit_els = []
            for compound in self.compounds:
                for el in Composition(compound).elements():
                    if el not in fit_els:
                        fit_els.append(el)

        # quick return if nothing to do
        if not len(fit_els):
            print 'Warning: Found no chemical potentials to fit'
            with open('mu_shifts.yaml', 'w') as outfile:
                yaml.dump({}, outfile, default_flow_style=False)
            return

        # gather exp form energy
        self.gather_fe(funcs=funcs, shift_mus=False, only_exp=True)

        # only keep compounds with at least 1 fitting el
        # not necessary if fit_els was All
        if not all_fit_els:
            self.filter_compounds(contains=[','.join(fit_els)])
            # self.print_compounds()

        # make sure we still have compounds
        if not len(self.compounds):
            sys.exit('Error: No compounds contain any of the fitting elements')

        # determine the unknown and known elements in the compound set
        all_els = []
        for compound in self.compounds:
            for el in Composition(compound).elements():
                if el not in all_els:
                    all_els.append(el)
        all_els = sort_systems(all_els)
        print '%3i Total Element(s):'%len(all_els), ','.join(sort_systems(all_els))
        unknown_els = sort_systems([el for el in fit_els if el in all_els]) # to fit
        known_els = sort_systems([el for el in all_els if el not in unknown_els]) # not to fit
        print '%3i     Element(s) to Fit:'%len(unknown_els), ','.join(sort_systems(unknown_els))
        if len(known_els): print '%3i Element(s) Not to Fit:'%len(known_els), ','.join(sort_systems(known_els))

        N = len(self.compounds)    # number of compounds
        M = len(unknown_els)       # number of unknown els
        L = len(known_els)         # number of known els

        X_k = np.zeros((N, L))     # fraction of known el in composition
        X_u = np.zeros((N, M))     # fraction of unknown el in composition
        Exp_fe = np.zeros((N, 1))  # compound experimental formation energy per atom

        for nn, compound in enumerate(sort_systems(self.compounds)):
            # stoichiometry matrices
            Compo = Composition(compound)
            for el in Compo.elements():
                if el in unknown_els:
                    X_u[nn, unknown_els.index(el)] = Compo.comp[el]
                elif el in known_els:
                    X_k[nn, known_els.index(el)] = Compo.comp[el]
                else: sys.exit('Error: Compound %s contains unknown element %s'%(compound, el))
            # compound exp formation energy per atom vector
            Exp_fe[nn] = np.mean([np.mean(self.fe[compound]['exp'][source]) for source in self.fe[compound]['exp']])

        # optimal mus contains all the chemical potentials (fitted and
        # non-fitted)
        optimal_mus = {el: {} for el in all_els}
        # mu corrections to be save to mu_shifts.yaml
        mu_shifts = {el: {} for el in unknown_els}
        for func in funcs:
            print 'Fitting mu for %s'%(func.upper())
            mu_k = np.zeros((L, 1))    # mu for known el
            # mu_u = np.zeros((M, 1))    # mu for unknown el
            Energy = np.zeros((N, 1))  # compound energy per atom

            for nn, compound in enumerate(sort_systems(self.compounds)):
                # compound energy per atom vector
                compound_folder = os.path.join(self.basedir, 'dft_runs', compound, func, 'rel_geom')
                Energy[nn] = self.total_energy(compound_folder)[1]

            # known mus
            for ll, el in enumerate(known_els):
                el_folder = os.path.join(self.basedir, 'dft_runs', el, func, 'rel_geom')
                mu_k[ll] = self.total_energy(el_folder)[1]
                optimal_mus[el][func] = float(mu_k[ll])

            AA = X_u
            bb = Energy - np.dot(X_k, mu_k) - Exp_fe
            fit = np.linalg.lstsq(AA, bb)

            mu_u = fit[0]

            if not quiet: print '%8s %14s'%('Element', 'Mu')
            for ll, el in enumerate(unknown_els):
                if not quiet: print '%8s % 14.3f'%(el, mu_u[ll])
                optimal_mus[el][func] = float(mu_u[ll])

            if not quiet: print '%8s %14s'%('Element', 'Mu Correction')
            for ll, el in enumerate(unknown_els):
                el_folder = os.path.join(self.basedir, 'dft_runs', el, func, 'rel_geom')
                unfitted_mu = self.total_energy(el_folder)[1]
                if not quiet: print '%8s % 14.3f'%(el, mu_u[ll]-unfitted_mu)
                mu_shifts[el][func] = float(mu_u[ll]-unfitted_mu)
            if not quiet: print

        # save mu data
        with open('optimal_mus.yaml', 'w') as outfile:
            yaml.dump(optimal_mus, outfile, default_flow_style=False)
        with open('mu_shifts.yaml', 'w') as outfile:
            yaml.dump(mu_shifts, outfile, default_flow_style=False)

    def cross_validation(self, funcs='', fit_els='All', include_zpe=False):
        '''Perform a cross-validation analysis for the mu fitting

        We perform 9-fold CV since 945 (number of compounds) is
        divisible by 9

        '''
        funcs = self._set_funcs(funcs)

        if len(self.compounds) != 945:
            sys.exit('Error: CV is currently hard coded for 945 compounds (not %i)'%(len(self.compounds)))

        # store original compounds
        orig_compounds = dict(self.compounds)

        # divide the compound set into 9 random groups
        compounds_random = list(self.compounds)
        shuffle(compounds_random)

        train_errors = {func: [] for func in funcs}
        test_errors = {func: [] for func in funcs}
        train_errors_mae = {func: [] for func in funcs}
        test_errors_mae = {func: [] for func in funcs}
        for ss in range(9):
            stats_dict = {} # testing
            test_compound_list = list(compounds_random[105*ss:105*(ss+1)])
            test_compounds_dict = {compound: orig_compounds[compound] for compound in test_compound_list}
            train_compound_list = [compound for compound in compounds_random if compound not in test_compound_list]
            train_compounds_dict = {compound: orig_compounds[compound] for compound in train_compound_list}
            print 'Subset %i'%ss

            # train
            self.compounds = dict(train_compounds_dict)
            self.gen_elements()
            self.gen_systems()
            print 'Training on %i compounds'%(len(self.compounds))
            self.chemical_potential_fitting(funcs=funcs, fit_els=fit_els, quiet=True)
            self.gather_fe(include_zpe=include_zpe, shift_mus=True, from_scratch=True)
            stats_dict = self.stats(ret=True, quiet=True)
            for func in funcs:
                train_errors[func].append(stats_dict[func][2]) # rmse
                train_errors_mae[func].append(stats_dict[func][1]) # mae
            # test
            print 'Testing'
            self.compounds = dict(test_compounds_dict)
            self.gen_elements()
            self.gen_systems()
            print 'Testing on %i compounds'%(len(self.compounds))
            self.gather_fe(include_zpe=include_zpe, shift_mus=True, from_scratch=True)
            stats_dict = self.stats(ret=True, quiet=True)
            for func in funcs:
                test_errors[func].append(stats_dict[func][2]) # rmse
                test_errors_mae[func].append(stats_dict[func][1]) # mae

        print '%8s %17s %17s %17s'%('Func', 'Train RMSE', 'Test RMSE', '(eV/atom)')
        for func in funcs:
            print '%8s % 17.5f % 17.5f'%(func.upper(), np.mean(train_errors[func]), np.mean(test_errors[func]))
        print
        print '%8s %17s %17s %17s'%('Func', 'Train MAE', 'Test MAE', '(eV/atom)')
        for func in funcs:
            print '%8s % 17.5f % 17.5f'%(func.upper(), np.mean(train_errors_mae[func]), np.mean(test_errors_mae[func]))

        # reset to original systems
        self.compounds = dict(orig_compounds)
        self.gen_elements()
        self.gen_systems()

    def plot_fe(self, funcs='', sources=['iit', 'ssub'],
                avg_expt=False, seedname='form_energies',
                plot_per_anion=False):
        '''Plot the formation energies'''
        if not len(self.fe): return
        funcs = self._set_funcs(funcs)
        # Print out the data
        DAT = open(os.path.join(self.basedir, seedname+'.dat'), 'w')
        DAT.write('# compound | calc_value (eV/atom)| exp_value (eV/atom) | label\n')
        DAT.write('# ---legend---\n')
        if plot_per_anion:
            DATPA = open(os.path.join(self.basedir, seedname+'_peranion.dat'), 'w')
            DATPA.write('# compound | calc_value (eV/atom)| exp_value (eV/atom) | label\n')
            DATPA.write('# ---legend---\n')
        DATERR = open(os.path.join(self.basedir, seedname+'_errors.dat'), 'w')
        DATERR.write('# compound | calc_value (eV/atom)| exp_value (eV/atom) | label\n')
        DATERR.write('# ---legend---\n')
        DATRELERR = open(os.path.join(self.basedir, seedname+'_relative_errors.dat'), 'w')
        DATRELERR.write('# compound | calc_value (eV/atom)| exp_value (eV/atom) | label\n')
        DATRELERR.write('# ---legend---\n')
        DATSAMELINE = open(os.path.join(self.basedir, seedname+'_sameline.dat'), 'w')
        DATSAMELINE.write('# compound | %s | exp_value (eV/atom) | label\n'%(' | '.join(['%4s_value (eV/atom)'%func for func in funcs])))
        DATSAMELINE.write('# ---legend---\n')
        hist_data = []
        for i in range(len(funcs+sources)):
            hist_data.append([])
        histerr_data = [] ; histrelerr_data = []
        for i in range(len(funcs)*len(sources)):
            histerr_data.append([])
            histrelerr_data.append([])
        HIST = open(os.path.join(self.basedir, seedname+'_histogram.dat'), 'w')
        HIST.write('# form_energy (eV/atom) | counts columns\n')
        HIST.write('# ---legend---\n')
        HISTERR = open(os.path.join(self.basedir, seedname+'_errors_histogram.dat'), 'w')
        HISTERR.write('# form_energy_error (eV/atom) | counts columns\n')
        HISTERR.write('# ---legend---\n')
        HISTRELERR = open(os.path.join(self.basedir, seedname+'_relative_errors_histogram.dat'), 'w')
        HISTRELERR.write('# relative_form_energy_error (%) | counts columns\n')
        HISTRELERR.write('# ---legend---\n')
        for pp, func_or_source in enumerate(funcs+sources):
            HIST.write('# ' + '%3i'%(pp+1) + ': ' + func_or_source + '\n')
        HIST.write('# ' + '%3i'%(len(funcs)*len(sources)+1) + ': ' + 'total exp\n')
        for comp in self.fe:
            if 'calc' in self.fe[comp] and 'exp' in self.fe[comp]:
                # only consider if have at least 1 calc and 1 exp value
                for pp, func_or_source in enumerate(funcs+sources):
                    calc_or_exp = 'calc' if func_or_source in funcs else 'exp'
                    if func_or_source in self.fe[comp][calc_or_exp]:
                        val = self.fe[comp][calc_or_exp][func_or_source]
                        hist_data[pp].append(val)
        for pp, prod in enumerate(product(funcs, sources)):
            DAT.write('# ' + '%3i'%(pp+1) + ': ' + ' '.join(prod) + '\n')
            if plot_per_anion:
                DATPA.write('# ' + '%3i'%(pp+1) + ': ' + ' '.join(prod) + '\n')
            HISTERR.write('# ' + '%3i'%(pp+1) + ': ' + ' '.join(prod) + '\n')
            HISTRELERR.write('# ' + '%3i'%(pp+1) + ': ' + ' '.join(prod) + '\n')
        for ss, source in enumerate(sources):
            DATSAMELINE.write('# %i %4s\n'%(ss+1, source))
        for pp, prod in enumerate(product(funcs)):
            HISTERR.write('# ' + '%3i'%(len(funcs)*len(sources)+pp+1) + ': ' + ' '.join(prod) + ' total\n')
            HISTRELERR.write('# ' + '%3i'%(len(funcs)*len(sources)+pp+1) + ': ' + ' '.join(prod) + ' total\n')

        # separate routines to print data to correlate errors from
        # different functionals
        for comp in sort_systems(self.fe):
            if 'calc' in self.fe[comp] and 'exp' in self.fe[comp]:
                for ss, source in enumerate(sources):
                    if source in self.fe[comp]['exp']:
                        if len(self.fe[comp]['exp'][source]) == 1:
                            data_row = ['%20s'%(gp_name(comp)), ' '.join(['% 9.3f'%val for val in self.fe[comp]['calc'].values()]), '% 9.3f'%(self.fe[comp]['exp'][source][0]), '%2d'%(ss+1)]
                            DATSAMELINE.write(' '.join(data_row) + '\n')
                        elif avg_expt:
                            data_row = ['%20s'%(gp_name(comp)), ' '.join(['% 9.3f'%val for val in self.fe[comp]['calc'].values()]), '% 9.3f'%(np.mean(self.fe[comp]['exp'][source])), '%2d'%(ss+1)]
                            DATSAMELINE.write(' '.join(data_row) + '\n')
                        else:
                            for same_db_val in self.fe[comp]['exp'][source]:
                                data_row = ['%20s'%(gp_name(comp)), ' '.join(['% 9.3f'%val for val in self.fe[comp]['calc'].values()]), '% 9.3f'%(same_db_val), '%2d'%(ss+1)]
                                DATSAMELINE.write(' '.join(data_row) + '\n')
        DATSAMELINE.close()

        min_exp = 999. ; max_exp = -999. ; min_calc = 999. ; max_calc = -999.
        min_err = 999. ; max_err = -999. ; min_rel_err = 999. ; max_rel_err = -999.
        for comp in sort_systems(self.fe):
            if 'calc' in self.fe[comp] and 'exp' in self.fe[comp]:
                # only can plot if have at least 1 calc and 1 exp value
                for pp, prod in enumerate(product(funcs, sources)):
                    if prod[0] in self.fe[comp]['calc'] and prod[1] in self.fe[comp]['exp']:
                        # check if there is another close point
                        energy_distances = [] ; energy_err_distances = [] ; energy_rel_err_distances = []
                        for comp2 in self.fe:
                            if 'calc' in self.fe[comp2] and 'exp' in self.fe[comp2]:
                                for pp2, prod2 in enumerate(product(funcs, sources)):
                                    if prod2[0] in self.fe[comp2]['calc'] and prod2[1] in self.fe[comp2]['exp']:
                                        energy_distance = np.sqrt((self.fe[comp]['calc'][prod[0]]-self.fe[comp2]['calc'][prod2[0]])**2 + (self.fe[comp]['exp'][prod[1]][0]-self.fe[comp2]['exp'][prod2[1]][0])**2)
                                        # adding factor of 2 for contribution from error (x) axis, since domain is roughly 1/2 the range
                                        energy_err_distance = np.sqrt(2*((self.fe[comp]['calc'][prod[0]]-self.fe[comp]['exp'][prod[1]][0])-(self.fe[comp2]['calc'][prod2[0]]-self.fe[comp2]['exp'][prod2[1]][0]))**2 + (self.fe[comp]['exp'][prod[1]][0]-self.fe[comp2]['exp'][prod2[1]][0])**2)
                                        # adding factor of 6 for contribution from rel error (x) axis
                                        energy_rel_err_distance = np.sqrt(6*((self.fe[comp]['calc'][prod[0]]-self.fe[comp]['exp'][prod[1]][0])/abs(self.fe[comp]['exp'][prod[1]][0])-(self.fe[comp2]['calc'][prod2[0]]-self.fe[comp2]['exp'][prod2[1]][0])/abs(self.fe[comp2]['exp'][prod2[1]][0]))**2 + (self.fe[comp]['exp'][prod[1]][0]-self.fe[comp2]['exp'][prod2[1]][0])**2)
                                        if energy_distance > 0: energy_distances.append(energy_distance)
                                        if energy_err_distance > 0: energy_err_distances.append(energy_err_distance)
                                        if energy_rel_err_distance > 0: energy_rel_err_distances.append(energy_rel_err_distance)
                        if min(energy_distances) < 0.12:
                            xoffs, yoffs = 0, 0 # too many nearby points
                        elif self.fe[comp]['calc'][prod[0]] < self.fe[comp]['exp'][prod[1]][0]:
                            xoffs, yoffs = -0.1, 0.1 # top left
                        elif self.fe[comp]['calc'][prod[0]] > self.fe[comp]['exp'][prod[1]][0]:
                            xoffs, yoffs = 0.1, -0.1 # bottom right
                        if min(energy_err_distances) < 0.15:
                            xoffs_err, yoffs_err = 0, 0 # too many nearby points
                        elif self.fe[comp]['calc'][prod[0]] < self.fe[comp]['exp'][prod[1]][0]:
                            xoffs_err, yoffs_err = -0.08, 0.08 # top left
                        elif self.fe[comp]['calc'][prod[0]] > self.fe[comp]['exp'][prod[1]][0]:
                            xoffs_err, yoffs_err = 0.08, 0.08 # top right
                        if min(energy_rel_err_distances) < 0.17:
                            xoffs_rel_err, yoffs_rel_err = 0, 0 # too many nearby points
                        elif self.fe[comp]['calc'][prod[0]] < self.fe[comp]['exp'][prod[1]][0]:
                            xoffs_rel_err, yoffs_rel_err = -4, 0.1 # top left
                        elif self.fe[comp]['calc'][prod[0]] > self.fe[comp]['exp'][prod[1]][0]:
                            xoffs_rel_err, yoffs_rel_err = 4, 0.1 # top right
                        if self.fe[comp]['calc'][prod[0]] > max_calc:
                            max_calc = self.fe[comp]['calc'][prod[0]]
                        elif self.fe[comp]['calc'][prod[0]] < min_calc:
                            min_calc = self.fe[comp]['calc'][prod[0]]
                        if len(self.fe[comp]['exp'][prod[1]]) == 1:
                            data_row = [gp_name(comp), self.fe[comp]['calc'][prod[0]], self.fe[comp]['exp'][prod[1]][0], pp+1]
                            if plot_per_anion:
                                anion_els = [anion_el for anion_el in ['H', 'N', 'O', 'F', 'Cl'] if Composition(comp).contains(anion_el)]
                                if len(anion_els) != 1:
                                    sys.exit('Error: Printing F.E. per anion element recypressed, but found != 1 diatomic molecule anion elements in %s'%comp)
                                else:
                                    anion_el = anion_els[0]
                                    data_row_pa = [gp_name(comp), self.fe[comp]['calc'][prod[0]]*(sum(Composition(comp).integer_comp.values())/float(Composition(comp).integer_comp[anion_el])), self.fe[comp]['exp'][prod[1]][0]*(sum(Composition(comp).integer_comp.values())/float(Composition(comp).integer_comp[anion_el])), pp+1]
                            err = float(self.fe[comp]['calc'][prod[0]])-float(self.fe[comp]['exp'][prod[1]][0])
                            rel_err = 100*(float(self.fe[comp]['calc'][prod[0]])-float(self.fe[comp]['exp'][prod[1]][0]))/abs(float(self.fe[comp]['exp'][prod[1]][0]))
                            histerr_data[pp].append(err)
                            histrelerr_data[pp].append(rel_err)
                            DAT.write(' '.join(['%20s'%str(dd) for dd in data_row[0:3]])+' % 8.2f % 8.2f '%(xoffs, yoffs)+'%2d'%(data_row[-1])+'\n')
                            if plot_per_anion:
                                DATPA.write(' '.join(['%20s'%str(dd) for dd in data_row_pa[0:3]])+' % 8.2f % 8.2f '%(xoffs, yoffs)+'%2d'%(data_row_pa[-1])+'\n')
                            DATERR.write(' '.join(['%20s'%str(dd) for dd in data_row[0:3]])+' % 8.2f % 8.2f '%(xoffs_err, yoffs_err)+'%2d'%(data_row[-1])+'\n')
                            DATRELERR.write(' '.join(['%20s'%str(dd) for dd in data_row[0:3]])+' % 8.2f % 8.2f '%(xoffs_rel_err, yoffs_rel_err)+'%2d'%(data_row[-1])+'\n')
                            if self.fe[comp]['exp'][prod[1]][0] > max_exp:
                                max_exp = self.fe[comp]['exp'][prod[1]][0]
                            elif self.fe[comp]['exp'][prod[1]][0] < min_exp:
                                min_exp = self.fe[comp]['exp'][prod[1]][0]
                            if err > max_err:
                                max_err = err
                            elif err < min_err:
                                min_err = err
                            if rel_err > max_rel_err:
                                max_rel_err = rel_err
                            elif rel_err < min_rel_err:
                                min_rel_err = rel_err
                        elif avg_expt:
                            data_row = [gp_name(comp), self.fe[comp]['calc'][prod[0]], np.mean(self.fe[comp]['exp'][prod[1]]), pp+1]
                            if plot_per_anion:
                                data_row_pa = [gp_name(comp), self.fe[comp]['calc'][prod[0]]*(sum(Composition(comp).integer_comp.values())/float(Composition(comp).integer_comp[anion_el])), np.mean(self.fe[comp]['exp'][prod[1]])*(sum(Composition(comp).integer_comp.values())/float(Composition(comp).integer_comp[anion_el])), pp+1]
                            err = self.fe[comp]['calc']-np.mean(self.fe[comp]['exp'][prod[1]])
                            rel_err = 100*(self.fe[comp]['calc'][prod[0]]-np.mean(self.fe[comp]['exp'][prod[1]]))/abs(np.mean(self.fe[comp]['exp'][prod[1]]))
                            histerr_data[pp].append(err)
                            histrelerr_data[pp].append(rel_err)
                            DAT.write(' '.join(['%20s'%str(dd) for dd in data_row[0:3]])+' % 8.2f % 8.2f '%(xoffs, yoffs)+'%2d'%(data_row[-1])+'\n')
                            if plot_per_anion:
                                DATPA.write(' '.join(['%20s'%str(dd) for dd in data_row_pa[0:3]])+' % 8.2f % 8.2f '%(xoffs, yoffs)+'%2d'%(data_row_pa[-1])+'\n')
                            DATERR.write(' '.join(['%20s'%str(dd) for dd in data_row[0:3]])+' % 8.2f % 8.2f '%(xoffs_err, yoffs_err)+'%2d'%(data_row[-1])+'\n')
                            DATRELERR.write(' '.join(['%20s'%str(dd) for dd in data_row[0:3]])+' % 8.2f % 8.2f '%(xoffs_rel_err, yoffs_rel_err)+'%2d'%(data_row[-1])+'\n')
                            if np.mean(self.fe[comp]['exp'][prod[1]]) > max_exp:
                                max_exp = np.mean(self.fe[comp]['exp'][prod[1]])
                            elif np.mean(self.fe[comp]['exp'][prod[1]]) < min_exp:
                                min_exp = np.mean(self.fe[comp]['exp'][prod[1]])
                            if rel_err > max_rel_err:
                                max_rel_err = rel_err
                            elif rel_err < min_rel_err:
                                min_rel_err = rel_err
                        else:
                            for same_db_val in self.fe[comp]['exp'][prod[1]]:
                                data_row = [gp_name(comp), self.fe[comp]['calc'][prod[0]], same_db_val, pp+1]
                                if plot_per_anion:
                                    data_row_pa = [gp_name(comp), self.fe[comp]['calc'][prod[0]]*(sum(Composition(comp).integer_comp.values())/float(Composition(comp).integer_comp[anion_el])), same_db_val*(sum(Composition(comp).integer_comp.values())/float(Composition(comp).integer_comp[anion_el])), pp+1]
                                err = self.fe[comp]['calc'][prod[0]]-same_db_val
                                rel_err = 100*(self.fe[comp]['calc'][prod[0]]-same_db_val)/abs(same_db_val)
                                DAT.write(' '.join(['%20s'%str(dd) for dd in data_row[0:3]])+' % 8.2f % 8.2f '%(xoffs, yoffs)+'%2d'%(data_row[-1])+'\n')
                                if plot_per_anion:
                                    DATPA.write(' '.join(['%20s'%str(dd) for dd in data_row_pa[0:3]])+' % 8.2f % 8.2f '%(xoffs, yoffs)+'%2d'%(data_row_pa[-1])+'\n')
                                DATERR.write(' '.join(['%20s'%str(dd) for dd in data_row[0:3]])+' % 8.2f % 8.2f '%(xoffs_err, yoffs_err)+'%2d'%(data_row[-1])+'\n')
                                DATRELERR.write(' '.join(['%20s'%str(dd) for dd in data_row[0:3]])+' % 8.2f % 8.2f '%(xoffs_rel_err, yoffs_rel_err)+'%2d'%(data_row[-1])+'\n')
                                if same_db_val > max_exp:
                                    max_exp = same_db_val
                                elif same_db_val < min_exp:
                                    min_exp = same_db_val
                                if rel_err > max_rel_err:
                                    max_rel_err = rel_err
                                elif rel_err < min_rel_err:
                                    min_rel_err = rel_err
                            # for the histogram, we always average values if from same source
                            histerr_data[pp].append(self.fe[comp]['calc']-np.mean(self.fe[comp]['exp'][prod[1]]))
                            histrelerr_data[pp].append((self.fe[comp]['calc']-np.mean(self.fe[comp]['exp'][prod[1]]))/abs(np.mean(self.fe[comp]['exp'][prod[1]])))
        DAT.close()
        if plot_per_anion: DATPA.close()
        DATERR.close()
        DATRELERR.close()
        # uniform histogram grid
        edges = np.arange(-6, 2.05, 0.15)
        edgeserr = np.arange(-2, 2.05, 0.05)
        edgesrelerr = np.arange(-4000, 4000.2, 5)
        # tics at middle of each bin
        tics = [0.5*(edges[i]+edges[i+1]) for i in range(len(edges)-1)]
        ticserr = [0.5*(edgeserr[i]+edgeserr[i+1]) for i in range(len(edgeserr)-1)]
        ticsrelerr = [0.5*(edgesrelerr[i]+edgesrelerr[i+1]) for i in range(len(edgesrelerr)-1)]
        # 1 extra column for avg over diff expt sources
        hist_values = np.zeros((len(tics), len(funcs+sources)+1))
        # for err and relerr: for each functional, 1 extra column for avg over diff expt sources
        histerr_values = np.zeros((len(ticserr), len(funcs)*(len(sources)+1)))
        histrelerr_values = np.zeros((len(ticsrelerr), len(funcs)*(len(sources)+1)))
        maxerr_y = [] # max hist values for each functional
        maxrelerr_y = []
        for pp in range(len(funcs+sources)):
            hist_values[:, pp] = np.histogram(np.array(hist_data[pp]), bins=edges)[0]
        for pp, prod in enumerate(product(funcs, sources)):
            histerr_values[:, pp] = np.histogram(np.array(histerr_data[pp]), bins=edgeserr)[0]
            histrelerr_values[:, pp] = np.histogram(np.array(histrelerr_data[pp]), bins=edgesrelerr)[0]
        exp_total = np.zeros(len(tics))
        for pp, func_or_source in enumerate(funcs+sources):
            if func_or_source in sources:
                exp_total = exp_total + hist_values[:, pp]
        hist_values[:, -1] = exp_total
        for f, func in enumerate(funcs):
            funcerr_total = np.zeros(len(ticserr))
            funcrelerr_total = np.zeros(len(ticsrelerr))
            for pp, prod in enumerate(product(funcs, sources)):
                if prod[0] == func:
                    funcerr_total = funcerr_total + histerr_values[:, pp]
                    funcrelerr_total = funcrelerr_total + histrelerr_values[:, pp]
            maxerr_y.append(max(funcerr_total))
            maxrelerr_y.append(max(funcrelerr_total))
            histerr_values[:, len(funcs)*len(sources)+f] = funcerr_total
            histrelerr_values[:, len(funcs)*len(sources)+f] = funcrelerr_total
        hist_max_y = max(hist_values[:, 1:].flatten())
        hist_maxerr_y = max(maxerr_y) # max for all functionals
        hist_maxrelerr_y = max(maxrelerr_y)
        for ent in range(len(tics)):
            HIST.write("% .3f %20s\n"%(tics[ent], ' '.join(['%3i'%(int(hist_values[ent, pp])) for pp in range(len(funcs+sources)+1)])))
        for ent in range(len(ticserr)):
            HISTERR.write("% .3f %20s\n"%(ticserr[ent], ' '.join(['%3i'%(int(histerr_values[ent, pp])) for pp in range(len(funcs)*(len(sources)+1))])))
        for ent in range(len(ticsrelerr)):
            HISTRELERR.write("% .3f %20s\n"%(ticsrelerr[ent], ' '.join(['%3i'%(int(histrelerr_values[ent, pp])) for pp in range(len(funcs)*(len(sources)+1))])))
        HIST.close()
        HISTERR.close()
        HISTRELERR.close()

        # Generate the gnuplot scripts
        # Absolute values
        full_seedname = seedname
        gp_header = '''\
@EPS size 3,3
set output '%s.eps'
set key at graph 0.4, 0.9 spacing 2.5 box width 1 font ',10'
set border back lw 0.2
set xlabel '{/Symbol D}H_{calc} (eV/atom)'
set ylabel '{/Symbol D}H_{exp} (eV/atom)'
set linestyle 1 lw 2 pt 5 ps 0.50 lc rgb 'blue'
set linestyle 2 lw 2 pt 7 ps 0.50 lc rgb 'blue'
set linestyle 3 lw 2 pt 4 ps 0.25 lc rgb 'orange'
set linestyle 4 lw 2 pt 6 ps 0.25 lc rgb 'orange'
set linestyle 99 dashtype 2 lw 2
set label 'overbinds' at graph 0.04, 0.95 tc rgb 'black' font ',11' left
set label 'underbinds' at graph 0.97, 0.05 tc rgb 'black' font ',11' right
perfect(x) = x # x=y line
set xrange [%.2f:%.2f]
set yrange [%.2f:%.2f]
plot \
perfect(x) w l ls 99 notitle '{/Symbol D}H_{exp} = {/Symbol D}H_{calc}', \\
'''
        GNU = open(os.path.join(self.basedir, full_seedname + '.gnuplot'), 'w')
        GNU.write(gp_header%(full_seedname, min(min_calc, min_exp)-0.2, max(max_calc, max_exp)+0.2, min(min_calc, min_exp)-0.2, max(max_calc, max_exp)+0.2))
        for pp, prod in enumerate(product(funcs, sources)):
            GNU.write("'%s.dat' u 2:($6 == %i ? $3: 1/0) ls %i title '%s', \\\n"%(full_seedname, pp+1, pp+1, '%4s - %4s'%(prod[0].upper(), prod[1].upper())))
            GNU.write("'%s.dat' u 2:($6 == %i ? $3: 1/0):1 w labels hypertext point ls %i notitle, \\\n"%(full_seedname, pp+1, pp+1))
        GNU.write("'%s.dat' u ($2+$4):($3+$5):($4 == 0 ? '' : stringcolumn(1)) with labels font ',10' offset char 0,0 notitle\n"%seedname)
        GNU.write("@FIXBB\n")
        GNU.close()
        with cd(self.basedir):
            execute_local('gnuplot %s.gnuplot'%full_seedname)

        # Absolute values per anion
        full_seedname = seedname + '_peranion'
        gp_header = '''\
@EPS size 3,3
set output '%s.eps'
set key at graph 0.4, 0.9 spacing 2.5 box width 1 font ',10'
set border back lw 0.2
set xlabel '{/Symbol D}H_{calc} (eV/anion)'
set ylabel '{/Symbol D}H_{exp} (eV/anion)'
set linestyle 1 lw 2 pt 5 ps 0.50 lc rgb 'blue'
set linestyle 2 lw 2 pt 7 ps 0.50 lc rgb 'blue'
set linestyle 3 lw 2 pt 4 ps 0.25 lc rgb 'orange'
set linestyle 4 lw 2 pt 6 ps 0.25 lc rgb 'orange'
set linestyle 99 dashtype 2 lw 2
set label 'overbinds' at graph 0.04, 0.95 tc rgb 'black' font ',11' left
set label 'underbinds' at graph 0.97, 0.05 tc rgb 'black' font ',11' right
perfect(x) = x # x=y line
plot \
perfect(x) w l ls 99 notitle '{/Symbol D}H_{exp} = {/Symbol D}H_{calc}', \\
'''
        GNU = open(os.path.join(self.basedir, full_seedname + '.gnuplot'), 'w')
        GNU.write(gp_header%full_seedname)
        for pp, prod in enumerate(product(funcs, sources)):
            GNU.write("'%s.dat' u 2:($6 == %i ? $3: 1/0) ls %i title '%s', \\\n"%(full_seedname, pp+1, pp+1, '%4s - %4s'%(prod[0].upper(), prod[1].upper())))
            GNU.write("'%s.dat' u 2:($6 == %i ? $3: 1/0):1 w labels hypertext point ls %i notitle, \\\n"%(full_seedname, pp+1, pp+1))
        GNU.write("'%s.dat' u ($2+$4):($3+$5):($4 == 0 ? '' : stringcolumn(1)) with labels font ',10' offset char 0,0 notitle\n"%full_seedname)
        GNU.write("@FIXBB\n")
        GNU.close()
        with cd(self.basedir):
            execute_local('gnuplot %s.gnuplot'%full_seedname)

        # Error values
        full_seedname = seedname + '_errors'
        gp_header = '''\
@EPS size 4,3
set output '%s.eps'
set key outside bottom spacing 2.5
set border back lw 0.2
set xlabel '{/Symbol D}H_{calc} - {/Symbol D}H_{exp} (eV/atom)'
set ylabel '{/Symbol D}H_{exp} (eV/atom)'
set linestyle 1 lw 2 pt 5 ps 0.50 lc rgb 'blue'
set linestyle 2 lw 2 pt 7 ps 0.50 lc rgb 'blue'
set linestyle 3 lw 2 pt 4 ps 0.25 lc rgb 'orange'
set linestyle 4 lw 2 pt 6 ps 0.25 lc rgb 'orange'
set linestyle 99 dashtype 2 lw 2
set label 'overbinds' at graph 0.04, 0.05 tc rgb 'black' font ',11' left
set label 'underbinds' at graph 0.97, 0.05 tc rgb 'black' font ',11' right
set xrange [%.2f:%.2f]
set yrange [%.2f:%.2f]
set yzeroaxis ls 99
plot \
'''
        GNU = open(os.path.join(self.basedir, full_seedname + '.gnuplot'), 'w')
        GNU.write(gp_header%(full_seedname, -max([abs(min_err), abs(max_err)])-0.2, max([abs(min_err), abs(max_err)])+0.2, min_exp-0.2, max_exp+0.2))
        for pp, prod in enumerate(product(funcs, sources)):
            GNU.write("'%s.dat' u ($6 == %i ? $2-$3: 1/0):3 ls %i title '%s', \\\n"%(full_seedname, pp+1, pp+1, '%4s - %4s'%(prod[0].upper(), prod[1].upper())))
            GNU.write("'%s.dat' u ($6 == %i ? $2-$3: 1/0):3:1 w labels hypertext point ls %i notitle, \\\n"%(full_seedname, pp+1, pp+1))
        GNU.write("'%s.dat' u ($2-$3+$4):($3+$5):($4 == 0 ? '' : stringcolumn(1)) with labels font ',10' offset char 0,0 notitle\n"%full_seedname)
        GNU.write("@FIXBB\n")
        GNU.close()
        with cd(self.basedir):
            execute_local('gnuplot %s.gnuplot'%full_seedname)

        # Correlation of error values
        full_seedname = 'correlate_errors'
        gp_header = '''\
@EPS size 4,3
set output '%s.eps'
set key left spacing 2.5 box width 2 height 1.1
set size ratio -1
set border back lw 0.2
set xlabel '{/Symbol D}H_{%s} - {/Symbol D}H_{exp} (eV/atom)'
set ylabel '{/Symbol D}H_{%s} - {/Symbol D}H_{exp} (eV/atom)'
set linestyle 1 lw 2 pt 5 ps 0.50 lc rgb 'orange'
set linestyle 2 lw 2 pt 7 ps 0.50 lc rgb 'blue'
set linestyle 99 dashtype 2 lw 2
set label 'overbinds' at graph 0.34, 0.05 tc rgb 'black' font ',11' left
set label 'underbinds' at graph 0.67, 0.05 tc rgb 'black' font ',11' right
set label 'overbinds' at graph 0.02, graph 0.47 tc rgb 'black' font ',11' left
set label 'underbinds' at graph 0.02, graph 0.54 tc rgb 'black' font ',11' left
set xrange [-1:1]
set yrange [-1:1]
set xzeroaxis ls 99
set yzeroaxis ls 99
# stats 'form_energies_sameline.dat' u ($2-$4):($3-$4)
plot \
'''
        GNU = open(os.path.join(self.basedir, full_seedname + '.gnuplot'), 'w')
        GNU.write(gp_header%(full_seedname, funcs[0].upper(), funcs[1].upper()))
        for ss, source in enumerate(sources):
            GNU.write("'%s.dat' u ($5 == %i ? $2-$%i: 1/0):($3-$%i) ls %i title '%s'%s"%('form_energies_sameline', ss+1, len(funcs)+2, len(funcs)+2, ss+1, '%4s'%(source.upper()), ', \\\n' if ss != len(sources)-1 else '\n'))
        GNU.write("@FIXBB\n")
        GNU.close()
        with cd(self.basedir):
            execute_local('gnuplot %s.gnuplot'%full_seedname)

        # Relative error values
        full_seedname = seedname + '_relative_errors'
        gp_header = '''\
@EPS size 4,3
set output '%s.eps'
set key outside bottom spacing 2.5
set border back lw 0.2
set xlabel '({/Symbol D}H_{calc} - {/Symbol D}H_{exp})/|{/Symbol D}H_{exp}| (%%)'
set ylabel '{/Symbol D}H_{exp} (eV/atom)'
set linestyle 1 lw 2 pt 5 ps 0.50 lc rgb 'blue'
set linestyle 2 lw 2 pt 7 ps 0.50 lc rgb 'blue'
set linestyle 3 lw 2 pt 4 ps 0.25 lc rgb 'orange'
set linestyle 4 lw 2 pt 6 ps 0.25 lc rgb 'orange'
set linestyle 99 dashtype 2 lw 2
set label 'overbinds' at graph 0.04, 0.05 tc rgb 'black' font ',11' left
set label 'underbinds' at graph 0.97, 0.05 tc rgb 'black' font ',11' right
set xrange [%.2f:%.2f]
# set xrange [-150:150]
set yrange [%.2f:%.2f]
set yzeroaxis ls 99
plot \
'''
        GNU = open(os.path.join(self.basedir, full_seedname + '.gnuplot'), 'w')
        GNU.write(gp_header%(full_seedname, -max([abs(min_rel_err), abs(max_rel_err)])-5., max([abs(min_rel_err), abs(max_rel_err)])+5., min_exp-0.2, max_exp+0.2))
        for pp, prod in enumerate(product(funcs, sources)):
            GNU.write("'%s.dat' u ($6 == %i ? 100*($2-$3)/abs($3): 1/0):3 ls %i title '%s', \\\n"%(full_seedname, pp+1, pp+1, '%4s - %4s'%(prod[0].upper(), prod[1].upper())))
            GNU.write("'%s.dat' u ($6 == %i ? 100*($2-$3)/abs($3): 1/0):3:1 w labels hypertext point ls %i notitle, \\\n"%(full_seedname, pp+1, pp+1))
        GNU.write("'%s.dat' u (100*(($2-$3)/abs($3))+$4):($3+$5):($4 == 0 ? '' : stringcolumn(1)) with labels font ',10' offset char 0,0 notitle\n"%full_seedname)
        GNU.write("@FIXBB\n")
        GNU.close()
        with cd(self.basedir):
            execute_local('gnuplot %s.gnuplot'%full_seedname)

        # Error histogram
        full_seedname = seedname + '_errors_histogram'
        gp_header = '''\
@EPS size 4,2
set output '%s.eps'
set key outside bottom spacing 2.5
set style fill solid border
set border back lw 0.2
set size ratio 0.5
set xlabel '{/Symbol D}H_{calc} - {/Symbol D}H_{exp} (eV/atom)'
set ylabel 'Number'
set linestyle 1 lw 0 lc rgb 'blue'
set linestyle 2 lw 0 lc rgb 'orange'
set linestyle 99 dashtype 2 lw 2 lc rgb 'black'
set label 'overbinds' at graph 0.04, 0.05 tc rgb 'black' font ',11' left
set label 'underbinds' at graph 0.97, 0.05 tc rgb 'black' font ',11' right
set xrange [-1:1]
YMIN=-%i
YMAX=%i
set yrange [YMIN:YMAX]
set arrow from 0,YMIN to 0,YMAX ls 99 nohead front
plot \
'''
        GNU = open(os.path.join(self.basedir, full_seedname + '.gnuplot'), 'w')
        GNU.write(gp_header%(full_seedname, hist_maxerr_y+1, hist_maxerr_y+1))
        for pp, prod in enumerate(product(funcs)):
            GNU.write("'%s.dat' u 1:(%s$%i) w boxes ls %i fs %s title '%s'%s"%(full_seedname, '-' if pp == 1 else '', pp+2+len(funcs)*len(sources), pp+1, 'solid' if pp == 0 else 'pattern 1', prod[0].upper(), '\n' if pp == len(funcs) - 1 else ', \\\n'))
        GNU.write("@FIXBB\n")
        GNU.close()
        with cd(self.basedir):
            execute_local('gnuplot %s.gnuplot'%full_seedname)

        # Relative error histogram
        full_seedname = seedname + '_relative_errors_histogram'
        gp_header = '''\
@EPS size 4,2
set output '%s.eps'
set key outside bottom spacing 2.5
set style fill solid border
set border back lw 0.2
set size ratio 0.5
set xlabel '({/Symbol D}H_{calc} - {/Symbol D}H_{exp})/|{/Symbol D}H_{exp}| (%%)'
set ylabel 'Number'
set linestyle 1 lw 0 lc rgb 'blue'
set linestyle 2 lw 0 lc rgb 'orange'
set linestyle 99 dashtype 2 lw 2 lc rgb 'black'
set label 'overbinds' at graph 0.04, 0.05 tc rgb 'black' font ',11' left
set label 'underbinds' at graph 0.97, 0.05 tc rgb 'black' font ',11' right
# set xrange [-150:150]
YMIN=-%i
YMAX=%i
set yrange [YMIN:YMAX]
set arrow from 0,YMIN to 0,YMAX ls 99 nohead front
plot \
'''
        GNU = open(os.path.join(self.basedir, full_seedname + '.gnuplot'), 'w')
        GNU.write(gp_header%(full_seedname, hist_maxrelerr_y+1, hist_maxrelerr_y+1))
        for pp, prod in enumerate(product(funcs)):
            GNU.write("'%s.dat' u 1:(%s$%i) w boxes ls %i fs %s title '%s'%s"%(full_seedname, '-' if pp == 1 else '', pp+2+len(funcs)*len(sources), pp+1, 'solid' if pp == 0 else 'pattern 1', prod[0].upper(), '\n' if pp == len(funcs) - 1 else ', \\\n'))
        GNU.write("@FIXBB\n")
        GNU.close()
        with cd(self.basedir):
            execute_local('gnuplot %s.gnuplot'%full_seedname)

        # Histogram
        full_seedname = seedname + '_histogram'
        gp_header = '''\
@EPS size 4,2
set output '%s.eps'
set key outside bottom spacing 2.5
set style fill solid border
set border back lw 0.2
set size ratio 0.5
set xlabel '{/Symbol D}H (eV/atom)'
set ylabel 'Number'
set linestyle 1 lw 0 lc rgb 'blue'
set linestyle 2 lw 3 lc rgb 'orange'
set linestyle 99 dashtype 2 lw 2 lc rgb 'black'
set xrange [%.2f:%.2f]
YMIN=%i
YMAX=%i
set yrange [YMIN:YMAX]
set arrow from 0,YMIN to 0,YMAX ls 99 nohead front
plot \\
'''
        GNU = open(os.path.join(self.basedir, full_seedname + '.gnuplot'), 'w')
        GNU.write(gp_header%(full_seedname, min([min_calc, min_exp])-0.2, max(max_calc, max_exp)+0.2, -(hist_max_y+2), hist_max_y+2))
        for pp, func in enumerate(funcs):
            col = '$%i'%(pp+2)
            GNU.write("'%s.dat' u 1:(%s) w boxes ls %i fs %s title '%s', \\\n"%(full_seedname, col, pp+1, 'solid' if pp == 0 else 'empty', func.upper()))
        for pp in range(len(sources)):
            col = '('+'+'.join(['$%i'%cl for cl in range(len(funcs)+len(sources)+1, len(funcs)+pp+1, -1)])+')'
            GNU.write("'%s.dat' u 1:(%s%s) w boxes ls %i fs pattern %i title '%s'%s"%(full_seedname, '-' if func_or_source in sources else '', col, len(funcs)+1+pp, pp+1, sources[pp].upper(), '\n' if pp == len(sources)-1 else ', \\\n'))
        GNU.write("@FIXBB\n")
        GNU.close()
        with cd(self.basedir):
            execute_local('gnuplot %s.gnuplot'%full_seedname)

    def stats(self, funcs='', sources=['iit', 'ssub'], avg_expt=True, compute_per_anion=False, ret=False, quiet=False):
        '''Compute statistics on the formation energy errors'''
        if not len(self.fe):
            print 'Warning: No stats computed since self.fe is empty'
            return
        funcs = self._set_funcs(funcs)
        stats_data = '''\
# Column number | Error type | Error
# Errors in eV/atom; Relative errors in percent
1  "ME"     % 10.3f
2  "MAE"    % 10.3f
3  "RMSE"   % 10.3f
4  "MRE"    % 10.1f
5  "MARE"   % 10.1f
6  "RMSRE"  % 10.1f
'''
        mag_err_vals = []
        mag_rel_err_vals = []
        if compute_per_anion:
            mag_err_vals_pa = []
            mag_rel_err_vals_pa = []
        # func: [me, mae, rmse, mre, mare, rmsre]
        stats_dict = {}
        for func in funcs:
            print 'Functional: %s'%func
            errors = [] ; rel_errors = [] ; max_abs_err = [0, 0, ''] ; max_abs_rel_err = [0, 0, '']
            if compute_per_anion: errors_pa = [] ; rel_errors_pa = [] ; max_abs_err_pa = [0, 0, ''] ; max_abs_rel_err_pa = [0, 0, '']
            for comp in self.fe:
                if compute_per_anion:
                    anion_els = [anion_el for anion_el in ['H', 'N', 'O', 'F', 'Cl'] if Composition(comp).contains(anion_el)]
                    if len(anion_els) != 1:
                        sys.exit('Error: Printing F.E. per anion element reed, but found != 1 diatomic molecule anion elements in %s'%comp)
                    else:
                        anion_el = anion_els[0]
                if 'calc' in self.fe[comp] and 'exp' in self.fe[comp]:
                    # only can include if have calc and at least 1 exp value
                    for pp, prod in enumerate(product([func], sources)):
                        if prod[0] in self.fe[comp]['calc'] and prod[1] in self.fe[comp]['exp']:
                            if len(self.fe[comp]['exp'][prod[1]]) == 1:
                                err = self.fe[comp]['calc'][prod[0]]-self.fe[comp]['exp'][prod[1]][0]
                                errors.append(err)
                                if abs(err) > max_abs_err[0]: max_abs_err = [abs(err), err, comp]
                                rel_err = 100*(self.fe[comp]['calc'][prod[0]]-self.fe[comp]['exp'][prod[1]][0])/abs(self.fe[comp]['exp'][prod[1]][0])
                                rel_errors.append(rel_err)
                                if abs(rel_err) > max_abs_rel_err[0]: max_abs_rel_err = [abs(rel_err), rel_err, comp]
                                if compute_per_anion:
                                    err_pa = (self.fe[comp]['calc'][prod[0]]-self.fe[comp]['exp'][prod[1]][0])*(sum(Composition(comp).integer_comp.values())/float(Composition(comp).integer_comp[anion_el]))
                                    errors_pa.append(err_pa)
                                    if abs(err_pa) > max_abs_err_pa[0]: max_abs_err_pa = [abs(err_pa), err_pa, comp]
                                    rel_err_pa = 100*(self.fe[comp]['calc'][prod[0]]-self.fe[comp]['exp'][prod[1]][0])/abs(self.fe[comp]['exp'][prod[1]][0])
                                    rel_errors_pa.append(rel_err_pa)
                                    if abs(rel_err_pa) > max_abs_rel_err_pa[0]: max_abs_rel_err_pa = [abs(rel_err_pa), rel_err_pa, comp]
                            elif avg_expt:
                                err = self.fe[comp]['calc'][prod[0]]-np.mean(self.fe[comp]['exp'][prod[1]][0])
                                errors.append(err)
                                if abs(err) > max_abs_err[0]: max_abs_err = [abs(err), err, comp]
                                rel_err = 100*(self.fe[comp]['calc'][prod[0]]-np.mean(self.fe[comp]['exp'][prod[1]][0]))/abs(np.mean(self.fe[comp]['exp'][prod[1]][0]))
                                rel_errors.append(rel_err)
                                if abs(rel_err) > max_abs_rel_err[0]: max_abs_rel_err = [abs(rel_err), rel_err, comp]
                                if compute_per_anion:
                                    err_pa = (self.fe[comp]['calc'][prod[0]]-np.mean(self.fe[comp]['exp'][prod[1]][0]))*(sum(Composition(comp).integer_comp.values())/float(Composition(comp).integer_comp[anion_el]))
                                    errors_pa.append(err_pa)
                                    if abs(err_pa) > max_abs_err_pa[0]: max_abs_err_pa = [abs(err_pa), err_pa, comp]
                                    rel_err_pa = 100*(self.fe[comp]['calc'][prod[0]]-np.mean(self.fe[comp]['exp'][prod[1]][0]))/abs(np.mean(self.fe[comp]['exp'][prod[1]][0]))
                                    rel_errors_pa.append(rel_err_pa)
                                    if abs(rel_err_pa) > max_abs_rel_err_pa[0]: max_abs_rel_err_pa = [abs(rel_err_pa), rel_err_pa, comp]
                            else:
                                for same_db_val in self.fe[comp]['exp'][prod[1]]:
                                    err = self.fe[comp]['calc'][prod[0]]-same_db_val
                                    errors.append(err)
                                    if abs(err) > max_abs_err[0]: max_abs_err = [abs(err), err, comp]
                                    rel_err = 100*(self.fe[comp]['calc'][prod[0]]-same_db_val)/abs(same_db_val)
                                    rel_errors.append(rel_err)
                                    if abs(rel_err) > max_abs_rel_err[0]: max_abs_rel_err = [abs(rel_err), rel_err, comp]
                                    if compute_per_anion:
                                        err_pa = (self.fe[comp]['calc'][prod[0]]-same_db_val)*(sum(Composition(comp).integer_comp.values())/float(Composition(comp).integer_comp[anion_el]))
                                        errors_pa.append(err_pa)
                                        if abs(err_pa) > max_abs_err_pa[0]: max_abs_err_pa = [abs(err_pa), err_pa, comp]
                                        rel_err_pa = 100*(self.fe[comp]['calc'][prod[0]]-same_db_val)/abs(same_db_val)
                                        rel_errors_pa.append(rel_err_pa)
                                        if abs(rel_err_pa) > max_abs_rel_err_pa[0]: max_abs_rel_err_pa = [abs(rel_err_pa), rel_err_pa, comp]
            me = (1./len(errors))*sum(errors)
            mae = (1./len(errors))*sum([abs(err) for err in errors])
            rmse = np.sqrt((1./len(errors))*sum([err**2 for err in errors]))
            mre = (1./len(rel_errors))*sum(rel_errors)
            mare = (1./len(rel_errors))*sum([abs(rel_err) for rel_err in rel_errors])
            rmsre = np.sqrt((1./len(rel_errors))*sum([rel_err**2 for rel_err in rel_errors]))
            mag_err_vals.extend([abs(me), abs(mae), abs(rmse)])
            mag_rel_err_vals.extend([abs(mre), abs(mare), abs(rmsre)])

            if not quiet:
                print '%42s'%'Mean Error (ME) =' + '% 10.3f eV/atom'%me
                print '%42s'%'Mean Average Error (MAE) =' + '% 10.3f eV/atom'%mae
                print '%42s'%'Root-Mean-Square Error (RMSE) =' + '% 10.3f eV/atom'%rmse
                print '%42s'%'Max Absolute Error =' + '% 10.3f eV/atom (%s)'%(max_abs_err[1], max_abs_err[2])
                print '%42s'%'Mean Relative Error (MRE) =' + '% 10.1f%%'%mre
                print '%42s'%'Mean Average Relative Error (MARE) =' + '% 10.1f%%'%mare
                print '%42s'%'Root-Mean-Square Relative Error (RMSRE) =' + '% 10.1f%%'%rmsre
                print '%42s'%'Max Absolute Relative Error =' + '% 10.1f%% (%s)'%(max_abs_rel_err[1], max_abs_rel_err[2])
                print
            DAT = open(func+'_stats.dat', 'w')
            DAT.write(stats_data%(me, mae, rmse, mre, mare, rmsre))
            DAT.close()

            if compute_per_anion:
                me_pa = (1./len(errors_pa))*sum(errors_pa)
                mae_pa = (1./len(errors_pa))*sum([abs(err) for err in errors_pa])
                rmse_pa = np.sqrt((1./len(errors_pa))*sum([err**2 for err in errors_pa]))
                mre_pa = (1./len(rel_errors_pa))*sum(rel_errors_pa)
                mare_pa = (1./len(rel_errors_pa))*sum([abs(rel_err) for rel_err in rel_errors_pa])
                rmsre_pa = np.sqrt((1./len(rel_errors_pa))*sum([rel_err**2 for rel_err in rel_errors_pa]))
                mag_err_vals_pa.extend([abs(me_pa), abs(mae_pa), abs(rmse_pa)])
                mag_rel_err_vals_pa.extend([abs(mre_pa), abs(mare_pa), abs(rmsre_pa)])

                if not quiet:
                    print '%42s'%'Mean Error (ME) =' + '% 10.3f eV/anion'%me_pa
                    print '%42s'%'Mean Average Error (MAE) =' + '% 10.3f eV/anion'%mae_pa
                    print '%42s'%'Root-Mean-Square Error (RMSE) =' + '% 10.3f eV/anion'%rmse_pa
                    print '%42s'%'Max Absolute Error =' + '% 10.3f eV/anion (%s)'%(max_abs_err_pa[1], max_abs_err_pa[2])
                    print '%42s'%'Mean Relative Error (MRE) =' + '% 10.1f%%'%mre_pa
                    print '%42s'%'Mean Average Relative Error (MARE) =' + '% 10.1f%%'%mare_pa
                    print '%42s'%'Root-Mean-Square Relative Error (RMSRE) =' + '% 10.1f%%'%rmsre_pa
                    print '%42s'%'Max Absolute Relative Error =' + '% 10.1f%% (%s)'%(max_abs_rel_err_pa[1], max_abs_rel_err_pa[2])
                    print
                DAT = open(func+'_stats_pa.dat', 'w')
                DAT.write(stats_data%(me_pa, mae_pa, rmse_pa, mre_pa, mare_pa, rmsre_pa))
                DAT.close()

            stats_dict[func] = list([me, mae, rmse, mre, mare, rmsre])

        gnu_header = '''\
@EPS size 4,3
set output 'stats.eps'

set size ratio 0.4
set xtics scale 0
set ytics scale 0.7
set y2tics scale 0.7
set ytics nomirror
set y2tics nomirror

set border back lw 0.3
set key at screen 0.56,0.8

set ylabel 'Error (meV/atom)'
set y2label 'Relative error (%%)'

set xrange [0.5:7]
set yrange [-%.1f:%.1f]
set y2range [-%.1f:%.1f]

set xzeroaxis lw 4
set xtics ("ME" 1.15, "MAE" 2.15, "RMSE" 3.15, "MRE" 4.15, "MARE" 5.15, "RMSRE" 6.15)

set arrow from first 3.625, graph 0 to first 3.625, graph 1 nohead dt 3
'''
        GNU = open('stats.gnuplot', 'w')
        GNU.write(gnu_header%(1100*max(mag_err_vals), 1100*max(mag_err_vals), 1.1*max(mag_rel_err_vals), 1.1*max(mag_rel_err_vals)))
        GNU.write('plot \\\n')
        for ff, func in enumerate(funcs):
            GNU.write("'./%s_stats.dat' u ($1 == 1 ? $1%s : 1/0):($3*1000):(0.3):1 axes x1y1 title '%s' w boxes %s lc variable, \\\n"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1)))
        for ff, func in enumerate(funcs):
            GNU.write("for [col=2:3] './%s_stats.dat' u ($1 == col ? $1%s : 1/0):($3*1000):(0.3):1 axes x1y1 notitle '%s' w boxes %s lc variable, \\\n"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1)))
        for ff, func in enumerate(funcs):
            GNU.write("for [col=4:6] './%s_stats.dat' u ($1 == col ? $1%s : 1/0):3:(0.3):1 axes x1y2 notitle '%s' w boxes %s lc variable%s"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1), '\n' if ff == len(funcs)-1 else ', \\\n'))
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local("gnuplot stats.gnuplot")

        gnu_header = '''\
@EPS size 4,3
set output 'stats_vertical.eps'

set size ratio 2
set xtics scale 0
set ytics scale 0.7
set y2tics scale 0.7
set ytics nomirror
set y2tics nomirror

set border back lw 0.3
set key at screen 0.25,0.95 font ',10'

set ylabel 'Error (meV/atom)'
set y2label 'Relative error (%%)'

set xrange [0.5:7]
set yrange [-%.1f:%.1f]
set y2range [-%.1f:%.1f]

set xzeroaxis lw 4
set xtics ("ME" 1.15, "MAE" 2.15, "RMSE" 3.15, "MRE" 4.15, "MARE" 5.15, "RMSRE" 6.15) font ',11'
set xtics rotate by -45
set xtics scale 1 out nomirror
set grid x lc rgb 'grey'

set arrow from first 3.625, graph 0 to first 3.625, graph 1 nohead dt 3
'''
        GNU = open('stats_vertical.gnuplot', 'w')
        GNU.write(gnu_header%(1100*max(mag_err_vals), 1100*max(mag_err_vals), 1.1*max(mag_rel_err_vals), 1.1*max(mag_rel_err_vals)))
        GNU.write('plot \\\n')
        for ff, func in enumerate(funcs):
            GNU.write("'./%s_stats.dat' u ($1 == 1 ? $1%s : 1/0):($3*1000):(0.3):1 axes x1y1 title '%s' w boxes %s lc variable, \\\n"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1)))
        for ff, func in enumerate(funcs):
            GNU.write("for [col=2:3] './%s_stats.dat' u ($1 == col ? $1%s : 1/0):($3*1000):(0.3):1 axes x1y1 notitle '%s' w boxes %s lc variable, \\\n"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1)))
        for ff, func in enumerate(funcs):
            GNU.write("for [col=4:6] './%s_stats.dat' u ($1 == col ? $1%s : 1/0):3:(0.3):1 axes x1y2 notitle '%s' w boxes %s lc variable%s"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1), '\n' if ff == len(funcs)-1 else ', \\\n'))
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local("gnuplot stats_vertical.gnuplot")

        if ret: return stats_dict

    def fe_mae(self, anion_mu_shift, anion_el, func, sources=['iit', 'ssub'], avg_expt=False):
        '''Compute the MAE for a particular anion chemical potential correction'''
        mag_err_vals = []
        mag_rel_err_vals = []

        errors = [] ; rel_errors = []
        for comp in self.fe:
            if not Composition(comp).contains(anion_el):
                sys.exit('Error: Found != 1 diatomic molecule anion elements in %s'%comp)
            if 'calc' in self.fe[comp] and 'exp' in self.fe[comp]:
                # only can include if have calc and at least 1 exp value
                for pp, prod in enumerate(product([func], sources)):
                    if prod[0] in self.fe[comp]['calc'] and prod[1] in self.fe[comp]['exp']:
                        # here we compute the calculated value assuming a particular mu shift
                        # need to subtract off the shift scaled by the fraction of the anion
                        computed_val = self.fe[comp]['calc'][prod[0]] - anion_mu_shift*((Composition(comp).integer_comp[anion_el])/float(sum(Composition(comp).integer_comp.values())))
                        if len(self.fe[comp]['exp'][prod[1]]) == 1:
                            err = computed_val-self.fe[comp]['exp'][prod[1]][0]
                            errors.append(err)
                            rel_err = 100*(computed_val-self.fe[comp]['exp'][prod[1]][0])/abs(self.fe[comp]['exp'][prod[1]][0])
                            rel_errors.append(rel_err)
                        elif avg_expt:
                            err = computed_val-np.mean(self.fe[comp]['exp'][prod[1]][0])
                            errors.append(err)
                            rel_err = 100*(computed_val-np.mean(self.fe[comp]['exp'][prod[1]][0]))/abs(np.mean(self.fe[comp]['exp'][prod[1]][0]))
                            rel_errors.append(rel_err)
                        else:
                            for same_db_val in self.fe[comp]['exp'][prod[1]]:
                                err = computed_val-same_db_val
                                errors.append(err)
                                rel_err = 100*(computed_val-same_db_val)/abs(same_db_val)
                                rel_errors.append(rel_err)
        me = (1./len(errors))*sum(errors)
        mae = (1./len(errors))*sum([abs(err) for err in errors])
        rmse = np.sqrt((1./len(errors))*sum([err**2 for err in errors]))
        mre = (1./len(rel_errors))*sum(rel_errors)
        mare = (1./len(rel_errors))*sum([abs(rel_err) for rel_err in rel_errors])
        rmsre = np.sqrt((1./len(rel_errors))*sum([rel_err**2 for rel_err in rel_errors]))
        mag_err_vals.extend([abs(me), abs(mae), abs(rmse)])
        mag_rel_err_vals.extend([abs(mre), abs(mare), abs(rmsre)])
        return mae

    def stats_volume(self, funcs='', include_elements=True):
        '''Compute statistics on the volume errors

        include_elements: whether to include elements in the
        statistics

        '''
        if not len(self.volume): return
        if not include_elements: print 'Warning: Elements not included in volume statistics'
        funcs = self._set_funcs(funcs)
        stats_data = '''\
# Column number | Error type | Error
# Errors in A^3/atom; Relative errors in percent
1  "ME"     % 10.3f
2  "MAE"    % 10.3f
3  "RMSE"   % 10.3f
4  "MRE"    % 10.1f
5  "MARE"   % 10.1f
6  "RMSRE"  % 10.1f
'''
        mag_err_vals = []
        mag_rel_err_vals = []
        for func in funcs:
            print 'Functional: %s'%func
            errors = [] ; rel_errors = [] ; max_abs_err = [0, 0, ''] ; max_abs_rel_err = [0, 0, '']
            for comp in self.volume:
                if include_elements or Composition(comp).nelements() != 1:
                    if comp in self.exp_volume and func in self.volume[comp] and 'rel_geom' in self.volume[comp][func]:
                        # only can include if have calc and exp value
                        err = self.volume[comp][func]['rel_geom']-self.exp_volume[comp]
                        errors.append(err)
                        if abs(err) > max_abs_err[0]: max_abs_err = [abs(err), err, comp]
                        rel_err = 100*(self.volume[comp][func]['rel_geom']-self.exp_volume[comp])/self.exp_volume[comp]
                        rel_errors.append(rel_err)
                        if abs(rel_err) > max_abs_rel_err[0]: max_abs_rel_err = [abs(rel_err), rel_err, comp]
            me = (1./len(errors))*sum(errors)
            mae = (1./len(errors))*sum([abs(err) for err in errors])
            rmse = np.sqrt((1./len(errors))*sum([err**2 for err in errors]))
            mre = (1./len(rel_errors))*sum(rel_errors)
            mare = (1./len(rel_errors))*sum([abs(rel_err) for rel_err in rel_errors])
            rmsre = np.sqrt((1./len(rel_errors))*sum([rel_err**2 for rel_err in rel_errors]))
            mag_err_vals.extend([abs(me), abs(mae), abs(rmse)])
            mag_rel_err_vals.extend([abs(mre), abs(mare), abs(rmsre)])

            print '%42s'%'Mean Error (ME) =' + '% 10.3f A^3/atom'%me
            print '%42s'%'Mean Average Error (MAE) =' + '% 10.3f A^3/atom'%mae
            print '%42s'%'Root-Mean-Square Error (RMSE) =' + '% 10.3f A^3/atom'%rmse
            print '%42s'%'Max Absolute Error =' + '% 10.3f A^3/atom (%s)'%(max_abs_err[1], max_abs_err[2])
            print '%42s'%'Mean Relative Error (MRE) =' + '% 10.1f%%'%mre
            print '%42s'%'Mean Average Relative Error (MARE) =' + '% 10.1f%%'%mare
            print '%42s'%'Root-Mean-Square Relative Error (RMSRE) =' + '% 10.1f%%'%rmsre
            print '%42s'%'Max Absolute Relative Error =' + '% 10.1f%% (%s)'%(max_abs_rel_err[1], max_abs_rel_err[2])
            print
            DAT = open(func+'_volume_stats.dat', 'w')
            DAT.write(stats_data%(me, mae, rmse, mre, mare, rmsre))
            DAT.close()

        gnu_header = '''\
@EPS size 4,3
set output 'volume_stats.eps'

set size ratio 0.4
set xtics scale 0
set ytics scale 0.7
set y2tics scale 0.7
set ytics nomirror
set y2tics nomirror

set border back lw 0.3
set key at screen 0.56,0.8

set ylabel 'Error (\\305^3/atom)'
set y2label 'Relative error (%%)'

set xrange [0.5:7]
set yrange [-%.1f:%.1f]
set y2range [-%.1f:%.1f]

set xzeroaxis lw 4
set xtics ("ME" 1.15, "MAE" 2.15, "RMSE" 3.15, "MRE" 4.15, "MARE" 5.15, "RMSRE" 6.15)

set arrow from first 3.625, graph 0 to first 3.625, graph 1 nohead dt 3
'''
        GNU = open('volume_stats.gnuplot', 'w')
        GNU.write(gnu_header%(1.1*max(mag_err_vals), 1.1*max(mag_err_vals), 1.1*max(mag_rel_err_vals), 1.1*max(mag_rel_err_vals)))
        GNU.write('plot \\\n')
        for ff, func in enumerate(funcs):
            GNU.write("'./%s_volume_stats.dat' u ($1 == 1 ? $1%s : 1/0):($3):(0.3):1 axes x1y1 title '%s' w boxes %s lc variable, \\\n"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1)))
        for ff, func in enumerate(funcs):
            GNU.write("for [col=2:3] './%s_volume_stats.dat' u ($1 == col ? $1%s : 1/0):($3):(0.3):1 axes x1y1 notitle '%s' w boxes %s lc variable, \\\n"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1)))
        for ff, func in enumerate(funcs):
            GNU.write("for [col=4:6] './%s_volume_stats.dat' u ($1 == col ? $1%s : 1/0):3:(0.3):1 axes x1y2 notitle '%s' w boxes %s lc variable%s"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1), '\n' if ff == len(funcs)-1 else ', \\\n'))
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local("gnuplot volume_stats.gnuplot")

        gnu_header = '''\
@EPS size 4,3
set output 'volume_stats_vertical.eps'

set size ratio 2
set xtics scale 0
set ytics scale 0.7
set y2tics scale 0.7
set ytics nomirror
set y2tics nomirror

set border back lw 0.3
set key at screen 0.25,0.95 font ',10'

set ylabel 'Error (\\305^3/atom)'
set y2label 'Relative error (%%)'

set xrange [0.5:7]
set yrange [-%.1f:%.1f]
set y2range [-%.1f:%.1f]

set xzeroaxis lw 4
set xtics ("ME" 1.15, "MAE" 2.15, "RMSE" 3.15, "MRE" 4.15, "MARE" 5.15, "RMSRE" 6.15) font ',11'
set xtics rotate by -45
set xtics scale 1 out nomirror
set grid x lc rgb 'grey'

set arrow from first 3.625, graph 0 to first 3.625, graph 1 nohead dt 3
'''
        GNU = open('volume_stats_vertical.gnuplot', 'w')
        GNU.write(gnu_header%(1.1*max(mag_err_vals), 1.1*max(mag_err_vals), 1.1*max(mag_rel_err_vals), 1.1*max(mag_rel_err_vals)))
        GNU.write('plot \\\n')
        for ff, func in enumerate(funcs):
            GNU.write("'./%s_volume_stats.dat' u ($1 == 1 ? $1%s : 1/0):($3):(0.3):1 axes x1y1 title '%s' w boxes %s lc variable, \\\n"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1)))
        for ff, func in enumerate(funcs):
            GNU.write("for [col=2:3] './%s_volume_stats.dat' u ($1 == col ? $1%s : 1/0):($3):(0.3):1 axes x1y1 notitle '%s' w boxes %s lc variable, \\\n"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1)))
        for ff, func in enumerate(funcs):
            GNU.write("for [col=4:6] './%s_volume_stats.dat' u ($1 == col ? $1%s : 1/0):3:(0.3):1 axes x1y2 notitle '%s' w boxes %s lc variable%s"%(func, '' if ff == 0 else '+%s'%(0.3*ff), func.upper(), 'fs solid' if ff == 0 else 'fill pattern %i'%(ff+1), '\n' if ff == len(funcs)-1 else ', \\\n'))
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local("gnuplot volume_stats_vertical.gnuplot")

    @for_each_calculation
    def _grab_volumes(self, system, func, jobtype):
        '''Grab the calculation volumes'''
        if system not in gas:
            if os.path.exists('OUTCAR') and os.path.exists('CONTCAR'):
                pos = Structure('CONTCAR')
                vol = pos.volume/pos.totalnatoms
                if system not in self.volume: self.volume[system] = {}
                if func not in self.volume[system]: self.volume[system][func] = {}
                self.volume[system][func][jobtype] = vol

    @for_each_calculation
    def _grab_gaps(self, system, func, jobtype, only_exp=True):
        '''Grab the calculation band gaps

        only_exp: only get if we have experimental value

        '''
        if not only_exp or system in self.exp_gaps:
            if os.path.exists('OUTCAR') and os.path.exists('DOSCAR'):
                gap = float(execute_local('rm -rf dos && ~/scripts/plotdos.sh && cd dos && get_gap.py dos.dat && cd ..')[0].split()[2])
                if system not in self.gaps: self.gaps[system] = {}
                self.gaps[system][func] = gap

    def _grab_exp_volumes(self):
        '''Grab the calculation volumes'''
        for system in self.systems:
            if system not in gas:
                if system == 'Ta':
                    # fix Ta since ICSD structure has hydrogen
                    # bcc w/ a=3.3029 from Edwards et al., JAP 22, 424 (1951)
                    self.exp_volume[system] = (3.3029**3)/2.
                    print 'Warning: Fixing %s volume since ICSD structure has hydrogen'%system
                elif system == 'Fe':
                    # fix Fe since ICSD structure is at high temperature
                    # bcc w/ a=2.855 from Davey, PR 25, 753 (1925)
                    self.exp_volume[system] = (2.855**3)/2.
                    print 'Warning: Fixing %s volume since ICSD structure is at high T'%system
                elif system == 'I':
                    # exclude I since ICSD structure is at high pressure
                    print 'Warning: Ignoring %s volume since ICSD structure is at high P'%system
                elif system == 'SiO2':
                    # SiO2 structure comes from DFT paper and no
                    # longer is in ICSD; we replace volume with that
                    # of duplicate entry 20578 (ICSD entry 162660)
                    print 'Warning: Fixing %s volume since ICSD structure was from DFT'%system
                    self.exp_volume[system] = 15.112328
                elif system == 'Zr2Ni':
                    # Zr2Ni structure comes from MD paper and no
                    # longer is in ICSD; we replace volume with that
                    # of duplicate entry 17865 (ICSD entry 105479)
                    print 'Warning: Fixing %s volume since ICSD structure was from MD'%system
                    self.exp_volume[system] = 18.322330
                elif system == 'CaSi':
                    # CaSi structure comes from DFT paper and no
                    # longer is in ICSD; we replace volume with that
                    # of duplicate entry 69648 (ICSD entry 78996)
                    print 'Warning: Fixing %s volume since ICSD structure was from DFT'%system
                    self.exp_volume[system] = 23.790717
                elif system == 'Ag2S':
                    # Ag2S structure comes from DFT paper and no
                    # longer is in ICSD; we replace volume with that
                    # from R. Sadanaga and S. Sueno, Miner. J. 5, 124
                    # (1967).
                    print 'Warning: Fixing %s volume since ICSD structure was from DFT'%system
                    self.exp_volume[system] = 18.953163
                elif system in self.exp_poscars:
                    pos = Structure(self.exp_poscars[system])
                    vol = pos.volume/pos.totalnatoms
                    self.exp_volume[system] = vol
                else: print 'Warning: No experimental structure for %s'%system

    @for_each_calculation
    def _grab_magnetism(self, system, func, jobtype, min_mom_mag):
        '''Grab the calculation magnetic moments'''
        if os.path.exists('OUTCAR') and os.path.exists('POSCAR'):
            pos = Structure('POSCAR')
            nions = pos.totalnatoms
            magmoms = execute_local("grep -A %i 'magnetization (x)' OUTCAR | tail -n %i"%(nions+3, nions))[0].splitlines()
            if len(magmoms) > 0:
                moms = [float(mag.split()[-1]) for mag in magmoms]
                if sum(moms) < 0: moms = [-mm for mm in moms] # should give all +ve magnetizations
                if any([abs(mm) > min_mom_mag for mm in moms]):
                    if system not in self.moms: self.moms[system] = {}
                    if func not in self.moms[system]: self.moms[system][func] = {}
                    self.moms[system][func][jobtype] = moms

    def grab_all_gaps(self, funcs=''):
        funcs = self._set_funcs(funcs)
        print '[Gathering all band gaps (saving to all_band_gaps.json)]'
        self._grab_gaps(funcs=funcs, jobtypes=['rel_geom'], only_exp=False)
        with open('all_band_gaps.json', 'w') as fp:
            json.dump(self.gaps, fp)

    def analyze_gaps(self, funcs=''):
        '''Analyze the band gaps of the calculations'''
        funcs = self._set_funcs(funcs)
        print '[Gathering, analyzing, and plotting band gaps]'
        self._grab_gaps(funcs=funcs, jobtypes=['rel_geom'])
        # write out one file per func
        for func in funcs:
            errors = []
            with open(func+'_gaps.dat', 'w') as f:
                f.write('%14s %14s %14s %24s\n'%('# System', '%4s (eV)'%(func.upper()), 'Exp (eV)', 'Closest Point Dist (eV)'))
                for system in sort_systems(self.exp_gaps):
                    if system in self.gaps:
                        if func in self.gaps[system]:
                            errors.append(self.gaps[system][func]-self.exp_gaps[system])
                            closest_pt = 99999.
                            for system2 in self.exp_gaps:
                                for func2 in funcs:
                                    if system2 != system or func2 != func:
                                        if system2 in self.gaps and func2 in self.gaps[system2]:
                                            # distance between 2 pts cannot be too close
                                            dist = np.sqrt((self.exp_gaps[system2]-self.exp_gaps[system])**2 + (self.gaps[system2][func2]-self.gaps[system][func])**2)
                                            if dist < closest_pt:
                                                if system2 != system or func != funcs[0]:
                                                    # ignore case of two different functionals for same system, except for one functional, to get a single label
                                                    closest_pt = dist
                            f.write('%12s %14.4f %14.4f %24.3f\n'%(gp_name(system), self.gaps[system][func], self.exp_gaps[system], closest_pt))
                        else:
                            f.write('%12s %14s %14.4f %24.3f\n'%(gp_name(system), '-', self.exp_gaps[system], 0.))
            mae = sum([abs(err) for err in errors])/len(errors)
            print 'Band gap MAE for %s is %.3f eV'%(func.upper(), mae)
        # generate the gnuplot file
        GNU = open('gaps.gnuplot', 'w')
        gnu_header = '''\
@EPS size 3,2
set output 'gaps.eps'
set key at graph 0.26, 0.8 spacing 2.5 box width 1.8 font ',10'
set border back lw 0.2
set xlabel '{/Symbol D}_{exp} (eV)'
set ylabel '{/Symbol D}_{calc} (eV)'
set linestyle 1 lw 2 pt 5 ps 0.7 lc rgb 'blue'
set linestyle 2 lw 2 pt 7 ps 0.7 lc rgb 'orange'
set linestyle 99 dashtype 2 lw 2
set label 'overestimate' at graph 0.04, 0.95 tc rgb 'black' font ',11' left
set label 'underestimate' at graph 0.99, 0.05 tc rgb 'black' font ',11' right
perfect(x) = x # x=y line
set xrange [0:13.7]
set yrange [0:11.5]
plot perfect(x) w l ls 99 notitle '{/Symbol D}_{calc} = {/Symbol D}_{exp}', \\
'''
        GNU.write(gnu_header)
        for ff, func in enumerate(funcs):
            GNU.write("'%s_gaps.dat' u 3:2 w p ls %i title '%s', \\\n"%(func, ff+1, func.upper()))
        for ff, func in enumerate(funcs):
            GNU.write("'%s_gaps.dat' u ($3+0):($2+0):($4 > 0.4 ? stringcolumn(1) : '') with labels font ',7' offset char 0.4,0.4 notitle%s"%(func, '\n' if ff == len(funcs)-1 else ', \\\n'))
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local('gnuplot gaps.gnuplot', error_ok=True)

    def analyze_volumes(self, funcs='', jobtypes=['relax', 'rel_geom'], include_elements=True):
        '''Analyze the volumes of the calculations'''
        funcs = self._set_funcs(funcs)
        print '[Gathering, analyzing, and plotting volumes]'
        if not include_elements: print 'Warning: Elements not included in volume statistics'
        self._grab_volumes(funcs=funcs, jobtypes=jobtypes)
        self._grab_exp_volumes()
        # calc avg volume for compounds for which all funcs done
        vol_averages = [0. for func in funcs]
        exp_vol_avg = 0.
        n_systems = 0 # number we average over
        n_exp_systems = 0 # number we average over
        for system in self.systems:
            if include_elements or Composition(system).nelements() != 1:
                if system in self.volume:
                    funcs_with_static = [func for func in funcs if func in self.volume[system] and 'rel_geom' in self.volume[system][func]]
                    if all(func in funcs_with_static for func in funcs):
                        n_systems += 1
                        # have static volume for all functionals
                        for ff, func in enumerate(funcs):
                            vol_averages[ff] += self.volume[system][func]['rel_geom']
                        if system in self.exp_volume:
                            n_exp_systems += 1
                            exp_vol_avg += self.exp_volume[system]
                        else:
                            print 'Warning: Not including %s for experimental volume'%system
        # normalize
        print '%8s %28s'%('Functional', 'Average Volume per Atom (A^3)')
        for ff, func in enumerate(funcs):
            vol_averages[ff] /= n_systems
            print '%8s % 28.2f'%(func, vol_averages[ff])
        exp_vol_avg /= n_exp_systems
        print '%8s % 28.2f'%('Exp', exp_vol_avg)

        # write out one file per func per jobtype
        outs = []
        for func in funcs:
            for jobtype in jobtypes:
                outs.append(open(func+'_'+jobtype+'_vols.dat', 'w'))
        out_exp = open('exp_vols.dat', 'w')
        for system in sort_systems(self.systems):
            if include_elements or Composition(system).nelements() != 1:
                if system in self.volume:
                    for fj, funcjobtype in enumerate(product(funcs, jobtypes)):
                        func = funcjobtype[0] ; jobtype = funcjobtype[1]
                        line = '%12s'%gp_name(system)
                        if func in self.volume[system] and jobtype in self.volume[system][func]:
                            line += '%12.4f'%self.volume[system][func][jobtype]
                        else:
                            line += '%12s'%('-')
                        outs[fj].write(line+'\n')
                    if system in self.exp_volume:
                        out_exp.write('%12s %12.4f\n'%(gp_name(system), self.exp_volume[system]))
                    else:
                        out_exp.write('%12s %12s\n'%(gp_name(system), '-'))
        for ot in range(len(outs)):
            outs[ot].close()
        out_exp.close()
        # generate the gnuplot file
        GNU = open('volume.gnuplot', 'w')
        gnu_header = '''\
@EPS size 70,3
set output 'volume.eps'
set size ratio 0.02
set border back lw 0.1
set xtics scale 0 rotate by -90 font ',8' nomirror offset 0, 0.1
set ytics scale 0.5
# set xlabel 'System'
set ylabel 'Volume (\\305^3/atom)'
set grid lc rgb 'grey'
set key outside
plot \\
'''
        GNU.write(gnu_header)
        for fj, funcjobtype in enumerate(product(funcs, jobtypes)):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_vols.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s %s', \\\n"%(func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), 'static' if jobtype == 'rel_geom' else jobtype))
        GNU.write("'exp_vols.dat' u 2:xticlabel(1) w p ps 1.0 lc 8 pt 9 title 'Experiment'\n")
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local('gnuplot volume.gnuplot', error_ok=True)
        GNU = open('volume_static.gnuplot', 'w')
        gnu_header = '''\
@EPS size 70,3
set output 'volume_static.eps'
set size ratio 0.02
set border back lw 0.1
set xtics scale 0 rotate by -90 font ',8' nomirror offset 0, 0.1
set ytics scale 0.5
# set xlabel 'System'
set ylabel 'Volume (\\305^3/atom)'
set grid lc rgb 'grey'
set key outside
plot \\
'''
        GNU.write(gnu_header)
        for fj, funcjobtype in enumerate(product(funcs, ['rel_geom'])):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_vols.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s', \\\n"%(func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper()))
        GNU.write("'exp_vols.dat' u 2:xticlabel(1) w p ps 1.0 lc 8 pt 9 title 'Experiment'\n")
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local('gnuplot volume_static.gnuplot', error_ok=True)

        GNU = open('volume_static_split.gnuplot', 'w')
        gnu_header = '''\
@EPS size 2*10,2*3.5
set output 'volume_static_split.eps'
set multiplot layout 3,1
set size ratio 0.1
set yrange [0:125]
set border back lw 0.5
set xtics rotate by -90 font '{/:Bold{,8}}' nomirror offset 0, 0.25
set ytics scale 1
set ytics 25
# set xlabel 'System'
set ylabel 'Volume (\\305^3/atom)' font ',35' offset -5
set grid x lc rgb 'grey'
set ytics format '%3.0f' font ',30'
set key top box width 3 height 0.5 horizontal font ',50' spacing 1.2
'''
        GNU.write(gnu_header)
        GNU.write('set xrange [0:350]\n')
        GNU.write('plot \\\n')
        for fj, funcjobtype in enumerate(product(funcs, ['rel_geom'])):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_vols.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s %s', \\\n"%(func, jobtype, 1.7 if fj%2 == 0 else 2.3, 4-fj, 7-fj, func.upper(), 'static' if jobtype == 'rel_geom' else jobtype))
        GNU.write("'exp_vols.dat' u 2:xticlabel(1) w p ps 2.0 lc 8 pt 9 title 'Experiment'\n")
        GNU.write("unset key\n")
        GNU.write("set yrange [0:60]\n")
        GNU.write("set ytics 20\n")
        GNU.write('set xrange [351:701]\n')
        GNU.write('plot \\\n')
        for fj, funcjobtype in enumerate(product(funcs, ['rel_geom'])):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_vols.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s %s', \\\n"%(func, jobtype, 1.7 if fj%2 == 0 else 2.3, 4-fj, 7-fj, func.upper(), 'static' if jobtype == 'rel_geom' else jobtype))
        GNU.write("'exp_vols.dat' u 2:xticlabel(1) w p ps 2.0 lc 8 pt 9 title 'Experiment'\n")
        GNU.write('set xrange [702:1053]\n')
        GNU.write('plot \\\n')
        for fj, funcjobtype in enumerate(product(funcs, ['rel_geom'])):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_vols.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s %s', \\\n"%(func, jobtype, 1.7 if fj%2 == 0 else 2.3, 4-fj, 7-fj, func.upper(), 'static' if jobtype == 'rel_geom' else jobtype))
        GNU.write("'exp_vols.dat' u 2:xticlabel(1) w p ps 2.0 lc 8 pt 9 title 'Experiment'\n")
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local('gnuplot volume_static_split.gnuplot', error_ok=True)

    def analyze_magnetism(self, funcs='', jobtypes=['relax', 'rel_geom'], min_mom_mag=0.1):
        '''Analyze the magnetism of the calculations'''
        print '[Gathering and plotting magnetic moments]'
        funcs = self._set_funcs(funcs)
        # gather the mom dict information
        max_atoms = 0
        self._grab_magnetism(funcs=funcs, jobtypes=jobtypes, min_mom_mag=min_mom_mag)
        # write out one file per func per jobtype
        outs = []
        # separate one for just rel_geom to avoid blank space from
        # relaxation ones
        only_static = []
        for func in funcs:
            for jobtype in jobtypes:
                outs.append(open(func+'_'+jobtype+'_moms.dat', 'w'))
            only_static.append(open(func+'_only_rel_geom_moms.dat', 'w'))
        for system in sort_systems(self.systems):
            if system in self.moms:
                for fj, funcjobtype in enumerate(product(funcs, jobtypes)):
                    func = funcjobtype[0] ; jobtype = funcjobtype[1]
                    line = '%12s'%gp_name(system)
                    if func in self.moms[system] and jobtype in self.moms[system][func]:
                        max_atoms = max(max_atoms, len(self.moms[system][func][jobtype]))
                        for mom in self.moms[system][func][jobtype]:
                            line += '%6.2f'%mom
                    else:
                        for at in range(len(self.moms[system][self.moms[system].keys()[0]][self.moms[system][self.moms[system].keys()[0]].keys()[0]])):
                            line += '%6s'%('-')
                    outs[fj].write(line+'\n')
                funcs_with_static = [func for func in funcs if func in self.moms[system] and 'rel_geom' in self.moms[system][func]]
                if len(funcs_with_static) > 0:
                    for ff, func in enumerate(funcs):
                        line = '%12s'%gp_name(system)
                        if func in self.moms[system] and 'rel_geom' in self.moms[system][func]:
                            for mom in self.moms[system][func]['rel_geom']:
                                line += '%6.2f'%mom
                        else:
                            for at in range(len(self.moms[system][funcs_with_static[0]]['rel_geom'])):
                                line += '%6s'%('-')
                        only_static[ff].write(line+'\n')
        # For each func, compute the max moment magnitude, averaged over compounds
        print 'Functional | Average Max-Absolute-Value Moment (mu_B) | Largest Moment (Compound)'
        for func in funcs:
            func_moms = []
            for system in self.moms:
                if all(fnc in self.moms[system] and 'rel_geom' in self.moms[system][fnc] for fnc in funcs):
                    # we have data for all functionals
                    func_moms.append([max([abs(mm) for mm in self.moms[system][func]['rel_geom']]), system])
            most_magnetic = max([ent[0] for ent in func_moms])
            print '%8s % 14.3f % 14.3f %15s'%(func, np.mean([ent[0] for ent in func_moms]), most_magnetic, [ent[1] for ent in func_moms if ent[0] == most_magnetic][0])

        for ot in range(len(outs)): outs[ot].close()
        for os in range(len(only_static)): only_static[os].close()
        # generate the gnuplot file
        GNU = open('magnetism.gnuplot', 'w')
        gnu_header = '''\
@EPS size 20,2
set output 'magnetism.eps'
set size ratio 0.05
set border back lw 0.1
set xtics scale 0 rotate by -90 font ',9' nomirror offset 0, 0.25
set ytics scale 0.5
# set xlabel 'System'
set ylabel 'Mag. Mom. ({/Symbol m}_B)'
set grid x lc rgb 'grey'
set key bottom box width 3 height 0.5 horizontal
set rmargin 10
set lmargin 10
set tmargin 2
set bmargin 2
plot \\
'''
        GNU.write(gnu_header)
        for fj, funcjobtype in enumerate(product(funcs, jobtypes)):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_moms.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s %s'%s"%(func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), 'static' if jobtype == 'rel_geom' else jobtype, '' if fj + 1 == len(funcs)*len(jobtypes) and max_atoms < 2 else ', \\\n'))
            if max_atoms > 1:
                GNU.write("for [i=3:%i] '%s_%s_moms.dat' u i:xticlabel(1) w p ps %.1f lc %i pt %i notitle '%s %s'%s"%(max_atoms+1, func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), 'static' if jobtype == 'rel_geom' else jobtype, '' if fj + 1 == len(funcs)*len(jobtypes) else ', \\\n'))
        GNU.write('\n')
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local('gnuplot magnetism.gnuplot', error_ok=True)
        GNU = open('magnetism_static.gnuplot', 'w')
        gnu_header = '''\
@EPS size 20,2
set output 'magnetism_static.eps'
set size ratio 0.05
set border back lw 0.1
set xtics scale 0 rotate by -90 font ',9' nomirror offset 0, 0.25
set ytics scale 0.5
# set xlabel 'System'
set ylabel 'Mag. Mom. ({/Symbol m}_B)'
set grid x lc rgb 'grey'
set key bottom box width 3 height 0.5 horizontal
set rmargin 10
set lmargin 10
set tmargin 2
set bmargin 2
plot \\
'''
        GNU.write(gnu_header)
        for fj, funcjobtype in enumerate(product(funcs, ['rel_geom'])):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_moms.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s'%s"%(func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) and max_atoms < 2 else ', \\\n'))
            if max_atoms > 1:
                GNU.write("for [i=3:%i] '%s_%s_moms.dat' u i:xticlabel(1) w p ps %.1f lc %i pt %i notitle '%s'%s"%(max_atoms+1, func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) else ', \\\n'))
        GNU.write('\n')
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local('gnuplot magnetism_static.gnuplot', error_ok=True)

        GNU = open('magnetism_only_static.gnuplot', 'w')
        gnu_header = '''\
@EPS size 20,2
set output 'magnetism_only_static.eps'
set size ratio 0.05
set border back lw 0.1
set xtics scale 0 rotate by -90 font ',9' nomirror offset 0, 0.25
set ytics scale 0.5
# set xlabel 'System'
set ylabel 'Mag. Mom. ({/Symbol m}_B)'
set grid x lc rgb 'grey'
set key bottom box width 3 height 0.5 horizontal
set rmargin 10
set lmargin 10
set tmargin 2
set bmargin 2
plot \\
'''
        GNU.write(gnu_header)
        for fj, funcjobtype in enumerate(product(funcs, ['only_rel_geom'])):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_moms.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s'%s"%(func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) and max_atoms < 2 else ', \\\n'))
            if max_atoms > 1:
                GNU.write("for [i=3:%i] '%s_%s_moms.dat' u i:xticlabel(1) w p ps %.1f lc %i pt %i notitle '%s'%s"%(max_atoms+1, func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) else ', \\\n'))
        GNU.write('\n')
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local('gnuplot magnetism_only_static.gnuplot', error_ok=True)

        GNU = open('magnetism_only_static_split.gnuplot', 'w')
        gnu_header = '''\
@EPS size 6,3.3
set output 'magnetism_only_static_split.eps'
set multiplot layout 3,1
set yrange [-5:5]
set size ratio 0.15
set border back lw 0.1
set xtics scale 0 rotate by -90 font ',9' nomirror offset 0, 0.25
set ytics scale 0.5
# set xlabel 'System'
set ylabel 'Mag. Mom. ({/Symbol m}_B)'
set grid x lc rgb 'grey'
set key bottom box width 3 height 0.5 horizontal
'''
        GNU.write(gnu_header)
        GNU.write('set xrange [0:66]\n')
        GNU.write('plot \\\n')
        for fj, funcjobtype in enumerate(product(funcs, ['only_rel_geom'])):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_moms.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s'%s"%(func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) and max_atoms < 2 else ', \\\n'))
            if max_atoms > 1:
                GNU.write("for [i=3:%i] '%s_%s_moms.dat' u i:xticlabel(1) w p ps %.1f lc %i pt %i notitle '%s'%s"%(max_atoms+1, func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) else ', \\\n'))
        GNU.write('\nunset key\n')
        GNU.write('set xrange [67:133]\n')
        GNU.write('plot \\\n')
        for fj, funcjobtype in enumerate(product(funcs, ['only_rel_geom'])):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_moms.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s'%s"%(func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) and max_atoms < 2 else ', \\\n'))
            if max_atoms > 1:
                GNU.write("for [i=3:%i] '%s_%s_moms.dat' u i:xticlabel(1) w p ps %.1f lc %i pt %i notitle '%s'%s"%(max_atoms+1, func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) else ', \\\n'))
        GNU.write('\nset xrange [134:200]\n')
        GNU.write('plot \\\n')
        for fj, funcjobtype in enumerate(product(funcs, ['only_rel_geom'])):
            func = funcjobtype[0] ; jobtype = funcjobtype[1]
            GNU.write("'%s_%s_moms.dat' u 2:xticlabel(1) w p ps %.1f lc %i pt %i title '%s'%s"%(func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) and max_atoms < 2 else ', \\\n'))
            if max_atoms > 1:
                GNU.write("for [i=3:%i] '%s_%s_moms.dat' u i:xticlabel(1) w p ps %.1f lc %i pt %i notitle '%s'%s"%(max_atoms+1, func, jobtype, 0.7 if fj%2 == 0 else 1.3, 4-fj, 7-fj, func.upper(), '' if fj + 1 == len(funcs) else ', \\\n'))
        GNU.write('\n')
        GNU.write("@FIXBB\n")
        GNU.close()
        execute_local('gnuplot magnetism_only_static_split.gnuplot', error_ok=True)

    def analyze_static_runtimes(self, funcs=''):
        '''Analyze the computational expense of the static calculations'''
        print '[Analyzing the computational cost of the calculations]'
        funcs = self._set_funcs(funcs)
        # average (over systems) enhancement of:
        # 1. number of electronic steps needed
        # 2. CPU-hours per electronic step
        # 3. total CPU-hours
        niter_enhancement = []
        tperiter_enhancement = []
        t_enhancement = []
        print '%20s %12s %3s %30s %8s'%('System', 'Functional', 'N_cores', 'T (s)', 'N_iter')
        for system in sort_systems(self.systems):
            # 1 list for PBE (assuming it is funcs[0]) and 1 for SCAN (funcs[1])
            # each list contains ncores, time, niter
            edat = []
            for ff, func in enumerate(funcs):
                folder = os.path.join(self.basedir, 'dft_runs', system, func, 'rel_geom')
                with cd(folder):
                    if os.path.isfile('OUTCAR'):
                        with open('OUTCAR', 'r') as f:
                            # grab the number of cores, number of electronic
                            # iterations, and the elapsed time
                            for line in f:
                                if 'total cores' in line:
                                    ncores = int(float(line.split()[2]))
                                elif 'Elapsed time (sec)' in line:
                                    time = float(line.split()[-1])
                                elif '-------------- Iteration' in line:
                                    niter = int(float(line.split('(')[-1].replace(')', '').replace(' ', '').replace('-', '')))
                        print '%20s %12s %3i %30.3f %8i'%(system, func.upper(), ncores, time, niter)
                        edat.append([ncores, time, niter])
            if len(edat) == 2:
                # additive enhancement
                # t_enhancement.append(((edat[1][0]*edat[1][1]) - (edat[0][0]*edat[0][1]))/(edat[0][0]*edat[0][1]))
                # niter_enhancement.append((edat[1][2]-edat[0][2])/edat[0][2])
                # tperiter_enhancement.append((((edat[1][0]*edat[1][1])/edat[1][2])-((edat[0][0]*edat[0][1])/edat[0][2]))/((edat[0][0]*edat[0][1])/edat[0][2]))

                # multiplicative enhancement
                t_enhancement.append((edat[1][0]*edat[1][1])/(edat[0][0]*edat[0][1]))
                niter_enhancement.append((edat[1][2])/edat[0][2])
                tperiter_enhancement.append(((edat[1][0]*edat[1][1])/edat[1][2])/((edat[0][0]*edat[0][1])/edat[0][2]))
                print '%100s % 10.4f'%(system, t_enhancement[-1])
            else: print 'Warning: Ignoring %s since not all static runs for %s were processed'%(system, ','.join(funcs))
        print 'Mean Enhancement | Mean Average Enhancement'
        print '%60s % 5.3f % 5.3f'%('No. of Electronic Iterations:', np.mean(niter_enhancement), np.mean([abs(val) for val in niter_enhancement]))
        print '%60s % 5.3f % 5.3f'%('CPU Time per Electronic Iteration:', np.mean(tperiter_enhancement), np.mean([abs(val) for val in tperiter_enhancement]))
        print '%60s % 5.3f % 5.3f'%('CPU Time:', np.mean(t_enhancement), np.mean([abs(val) for val in t_enhancement]))

if __name__ == '__main__':
    # Parse the input parameters
    inp = inputs("""basedir='.' compound_file='compounds.csv'
reference_file='reference_states.csv' pos_file='poscars.csv'
exp_pos_file='exp_poscars.csv' exp_gap_file='exp_bandgaps.csv'
visualize=str calc_settings_file='calc_settings.csv' fe_file='fe.csv'
cluster=str generate=str monitor=str cancel=str submit=str analyze=str
plot=str avg_fe_expt=True zpe=str ignore_oqmd_dftu=str magnetism=str
volume=str gaps=str filter_compounds='True' compounds_max_natoms=999
compounds_contain='' compounds_lack='' compounds_exp_fe=']0.05'
compounds_remove_ambiguous_exp_fe='True'
compounds_all_calcs_finished='True' only_compounds=str shift_mus=str
per_anion=str optimize_single_anion=str volume_compounds_only=str""")
    inp.update(sys.argv[1:])
    inp.report()

    # instantiate the class
    bb = Benchmark(basedir=os.path.abspath(inp.basedir))

    # compounds
    bb.compounds_from_file(filename=inp.compound_file)
    # bb.print_compounds()
    if dir(inp).count('generate'):
        bb.save_compounds()

    # elements
    if not dir(inp).count('only_compounds'):
        bb.gen_elements(ref_file=inp.reference_file)
        bb.print_elements()
        if dir(inp).count('generate'):
            bb.save_elements()

    # systems
    bb.gen_systems()

    # poscars
    if dir(inp).count('generate') and not os.path.isfile(inp.pos_file):
        bb.gen_poscars()
        bb.save_poscars(inp.pos_file, experimental=False)
    elif os.path.isfile(inp.pos_file):
        bb.read_poscars(inp.pos_file, experimental=False)

    # filter compounds
    if dir(inp).count('filter_compounds'):
        bb.filter_compounds(max_natoms=inp.compounds_max_natoms, contains=inp.compounds_contain.split(';'), lacks=inp.compounds_lack.split(','), all_calcs_finished=dir(inp).count('compounds_all_calcs_finished'))
        if inp.compounds_exp_fe != '':
            # need to grab the exp f.e. with which we filter
            # empty funcs set since we only want exp values
            bb.gather_fe(funcs=[])
            bb.filter_compounds(exp_fe=inp.compounds_exp_fe.split(';'), remove_ambiguous_exp_fe=dir(inp).count('compounds_remove_ambiguous_exp_fe'), all_calcs_finished=False)
        elif dir(inp).count('compounds_remove_ambiguous_exp_fe'):
            # case for removing ambiguous exp fe but not filtering
            # based on avg exp fe
            bb.gather_fe(funcs=[])
            bb.filter_compounds(remove_ambiguous_exp_fe=True, all_calcs_finished=False)
        bb.print_compounds()
        # regenerate elements (removes elements not included in the filtered set of compounds)
        bb.gen_elements(ref_file=inp.reference_file)
        bb.print_elements()
        # regenerate systems
        bb.gen_systems()

    # make sure we know the cluster
    if hasattr(inp, 'cluster'):
        if inp.cluster not in known_clusters:
            sys.exit('Error: %s cluster is not known'%inp.cluster)

    # calculate
    if inp.calc_settings_file:
        bb.read_calc_settings(filename=inp.calc_settings_file)
    if dir(inp).count('generate'):
        if hasattr(inp, 'cluster'):
            bb.gen_inputs(cluster=inp.cluster, **bb.calc_settings)
            bb.kpt_table(jobtype='relax')
        else: sys.exit('Error: cluster must be set to generate job scripts')

    # visualize
    if dir(inp).count('visualize'):
        bb.visualize(funcs=['pbe'])

    # monitor
    if dir(inp).count('monitor'):
        if hasattr(inp, 'cluster'):
            bb.check_relax(inp.cluster)
            bb.check_static(inp.cluster)
        else: sys.exit('Error: cluster must be set to generate any modified job scripts')

    # cancel
    if dir(inp).count('cancel'):
        if hasattr(inp, 'cluster'):
            bb.cancel_jobs(clusters=[inp.cluster], include_running_jobs=False)
        else: sys.exit('Error: cluster must be set to cancel jobs')

    # submit
    if dir(inp).count('submit'):
        if hasattr(inp, 'cluster'):
            bb.submit_jobs(inp.cluster)
        else: sys.exit('Error: cluster must be set to submit jobs')

    # bb.kpt_table(jobtype='rel_geom')

    # analyze
    if dir(inp).count('analyze'):
        bb.kpt_table(jobtype='rel_geom')
        inc_zpe = dir(inp).count('zpe')
        if inc_zpe: bb.gather_zpe()
        bb.gather_fe(include_zpe=inc_zpe, ignore_oqmd_dftu=dir(inp).count('ignore_oqmd_dftu'), shift_mus=dir(inp).count('shift_mus'))
        bb.save_fe(outfile=inp.fe_file, avg_expt=inp.avg_fe_expt)
        if dir(inp).count('optimize_single_anion'):
            bb.optimize_single_anion_mu()

    # plot
    if dir(inp).count('plot'):
        bb.read_fe(infile=inp.fe_file)
        bb.print_fe(avg_expt=False)
        bb.plot_fe(avg_expt=inp.avg_fe_expt, plot_per_anion=dir(inp).count('per_anion'))
        bb.stats(compute_per_anion=dir(inp).count('per_anion'))

    # magnetism
    if dir(inp).count('magnetism'):
        bb.analyze_magnetism()

    # volume
    if dir(inp).count('volume'):
        if os.path.isfile(inp.exp_pos_file):
            bb.read_poscars(inp.exp_pos_file, experimental=True)
        else:
            bb.grab_exp_poscars()
            bb.save_poscars(inp.exp_pos_file, experimental=True)
        bb.analyze_volumes(include_elements=not dir(inp).count('volume_compounds_only'))
        bb.stats_volume(include_elements=not dir(inp).count('volume_compounds_only'))

    # band gaps
    if dir(inp).count('gaps'):
        if os.path.isfile(inp.exp_gap_file):
            bb.read_gaps(inp.exp_gap_file)
            bb.analyze_gaps()
        else: sys.exit('Error: %s file for experimental band gaps does not exist'%inp.exp_gap_file)
