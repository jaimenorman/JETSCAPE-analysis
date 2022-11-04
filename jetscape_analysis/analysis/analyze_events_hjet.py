#!/usr/bin/env python3

"""
  Class to analyze a single JETSCAPE parquet output file,
  and write out a new parquet file containing calculated observables

  For AA, must perform hole subtraction:
    For hadron observables:
      We save spectra for positive/negative particles separately, then subtract at histogram-level in plotting script
    For jets we find three different collections of jets:
      (1) Using shower+recoil particles, with constituent subtraction
           - No further hole subtraction necessary
      (2) Using shower+recoil particles, with standard recombiner
          In this case, observable-specific hole subtraction necessary
          We consider three different classes of jet observables:
           (i) Jet pt-like observables -- subtract holes within R
           (ii) Additive substructure -- subtract holes within R
           (iii) Non-additive substructure -- correct the jet pt only
          We also save unsubtracted histograms for comparison (although for substructure we still correct pt)
      (3) Using shower+recoil+hole particles, with negative recombiner
          In this case, observable-specific hole subtraction necessary
          We consider three different classes of jet observables:
           (i) Jet pt-like observables -- no further hole subtraction
           (ii) Additive substructure -- subtract holes within R
           (iii) Non-additive substructure -- no further hole subtraction

  Author: James Mulligan (james.mulligan@berkeley.edu)
  Author: Raymond Ehlers (raymond.ehlers@cern.ch)
  """

from __future__ import print_function

# General
import sys
import os
import argparse
import yaml
import numpy as np
import random
import math
from collections import defaultdict
from pathlib import Path

# Fastjet via python (from external library heppy)
import fjcontrib
import fjext
import fastjet as fj

import ROOT

sys.path.append('.')
from jetscape_analysis.analysis import analyze_events_base_STAT

################################################################
class AnalyzeJetscapeEvents_STAT(analyze_events_base_STAT.AnalyzeJetscapeEvents_BaseSTAT):

    # ---------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------
    def __init__(self, config_file='', input_file='', output_dir='', **kwargs):
        super(AnalyzeJetscapeEvents_STAT, self).__init__(config_file=config_file,
                                                         input_file=input_file,
                                                         output_dir=output_dir,
                                                         **kwargs)
        # Initialize config file
        self.initialize_user_config()

        print(self)

    # ---------------------------------------------------------------
    # Initialize config file into class members
    # ---------------------------------------------------------------
    def initialize_user_config(self):

        # Read config file
        with open(self.config_file, 'r') as stream:
            config = yaml.safe_load(stream)

        self.sqrts = config['sqrt_s']
        self.output_file = f'observables'
        # Update the output_file to contain the labeling in the final_state_hadrons file.
        # We use this naming convention as the flag for whether we should attempt to rename it.
        if "final_state_hadrons" in self.input_file_hadrons:
            _input_filename = Path(self.input_file_hadrons).name
            # The filename will be something like "observables_0000_00.parquet", assuming
            # that the original name was "observables"
            self.output_file = _input_filename.replace("final_state_hadrons", self.output_file)
            #print(f'Updated output_file name to "{self.output_file}" in order to add identifying indices.')

        # Load observable blocks
        self.hadron_observables = config['hadron']
        self.hadron_correlation_observables = config['hadron_correlations']
        self.inclusive_chjet_observables = config['inclusive_chjet']
        self.inclusive_jet_observables = {}
        self.semi_inclusive_chjet_observables = {}
        self.dijet_observables = {}
        if 'inclusive_jet' in config:
            self.inclusive_jet_observables = config['inclusive_jet']
        if 'semi_inclusive_chjet' in config:
            self.semi_inclusive_chjet_observables = config['semi_inclusive_chjet']
        if 'dijet' in config:
            self.dijet_observables = config['dijet']

        # General jet finding parameters
        self.jet_R = config['jet_R']
        self.min_jet_pt = config['min_jet_pt']
        self.max_jet_y = config['max_jet_y']

        # General grooming parameters'
        self.grooming_settings = {}
        if 'SoftDrop' in config:
            self.grooming_settings = config['SoftDrop']

        # If AA, set different options for hole subtraction treatment
        if self.is_AA:
            self.jet_collection_labels = config['jet_collection_labels']
        else:
            self.jet_collection_labels = ['']

    # ---------------------------------------------------------------
    # Initialize output objects
    # ---------------------------------------------------------------
    def initialize_user_output_objects(self):

        # Hadron histograms
        hname = 'hChHadronPt'
        h = ROOT.TH1F(hname, hname, 100, 0, 100)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hPtHat'
        h = ROOT.TH1F(hname, hname, 100, 0, 500)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hVertexX'
        h = ROOT.TH1F(hname, hname, 201, -20.05, 20.05)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hVertexY'
        h = ROOT.TH1F(hname, hname, 201, -20.05, 20.05)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hVertexXY'
        h = ROOT.TH2F(hname, hname, 201, -20.05, 20.05, 201, -20.05, 20.05)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hSumVertexXYPtHat'
        h = ROOT.TH2F(hname, hname, 100, 0, 20, 50,0,500)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hSumVertexXYPtChHadron'
        h = ROOT.TH2F(hname, hname, 100, 0, 20, 50,0,100)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hPtHatPtChHadron'
        h = ROOT.TH2F(hname, hname, 50, 0, 500, 50,0,100)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hSumVertexXYPtHatPtChHadron'
        h = ROOT.TH3F(hname, hname, 100, 0, 20, 50, 0, 500, 50,0,100)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hSumVertexXYPtChHadronDirected'
        h = ROOT.TH2F(hname, hname, 100, 0, 20, 50,0,100)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hVertexXPtChHadronDirected'
        h = ROOT.TH2F(hname, hname, 201,-20.05, 20.05, 50,0,100)
        h.Sumw2()
        setattr(self, hname, h)

        hname = 'hVertexXPtChHadronRotate'
        h = ROOT.TH2F(hname, hname, 201,-20.05, 20.05, 50,0,100)
        h.Sumw2()
        setattr(self, hname, h)


        for jet_collection_label in self.jet_collection_labels:
            for jetR in self.jet_R:
                hname = f'hVertexXPtJetRotate_{jetR}{jet_collection_label}'
                h = ROOT.TH2F(hname, hname, 201,-20.05, 20.05, 150,0,300)
                h.Sumw2()
                setattr(self, hname, h)


    # ---------------------------------------------------------------
    # Analyze a single event -- fill user-defined output objects
    # ---------------------------------------------------------------
    def analyze_event(self, event):

        # Initialize a dictionary that will store a list of calculated values for each output observable
        self.observable_dict_event = defaultdict(list)

        # hadrons = event.hadrons()
        # self.fill_hadron_histograms(hadrons)
        # print(event.__dict__)

        # Create list of fastjet::PseudoJets (separately for jet shower particles and holes)
        fj_hadrons_positive, pid_hadrons_positive = self.fill_fastjet_constituents(event, select_status='+')
        fj_hadrons_negative, pid_hadrons_negative = self.fill_fastjet_constituents(event, select_status='-')

        # Create list of charged particles
        fj_hadrons_positive_charged, pid_hadrons_positive_charged = self.fill_fastjet_constituents(event, select_status='+',
                                                                     select_charged=True)
        fj_hadrons_negative_charged, pid_hadrons_negative_charged = self.fill_fastjet_constituents(event, select_status='-',
                                                                     select_charged=True)


        self.fill_hadron_histograms_fj(fj_hadrons_positive_charged, event)


        # Fill jet observables
        for jet_collection_label in self.jet_collection_labels:
            # If constituent subtraction, subtract the event (with rho determined from holes) -- we can then neglect the holes
            if jet_collection_label == '_constituent_subtraction':
                self.bge_rho.set_particles(fj_hadrons_negative)
                hadrons_positive = self.constituent_subtractor.subtract_event(fj_hadrons_positive)
                hadrons_negative = None

                self.bge_rho.set_particles(fj_hadrons_negative_charged)
                hadrons_positive_charged = self.constituent_subtractor.subtract_event(fj_hadrons_positive_charged)
                hadrons_negative_charged = None

            # For shower_recoil and negative_recombiner cases, keep both positive and negative hadrons
            else:
                hadrons_positive = fj_hadrons_positive
                hadrons_negative = fj_hadrons_negative
                hadrons_positive_charged = fj_hadrons_positive_charged
                hadrons_negative_charged = fj_hadrons_negative_charged

            # Find jets and fill observables
            self.fill_jet_observables(event, hadrons_positive, hadrons_negative,
                                      hadrons_positive_charged, hadrons_negative_charged,
                                      pid_hadrons_positive, pid_hadrons_negative,
                                      pid_hadrons_positive_charged, pid_hadrons_negative_charged,
                                      jet_collection_label=jet_collection_label)




    # ---------------------------------------------------------------
    # Fill hadron histograms
    # ---------------------------------------------------------------
    def fill_hadron_histograms(self, hadrons):
    
        # Loop through hadrons
        for hadron in hadrons:

            # Fill some basic hadron info
            pid = hadron.pid
            pt = hadron.momentum.pt()

            # Fill charged hadron histograms (pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
            if abs(pid) in [211, 321, 2212, 3222, 3112, 3312, 3334]:
                getattr(self, 'hChHadronPt').Fill(pt, 1/pt) # Fill with weight 1/pt, to form 1/pt dN/dpt

    # ---------------------------------------------------------------
    # Fill hadron histograms with fastjet hadrons
    # ---------------------------------------------------------------
    def fill_hadron_histograms_fj(self, hadrons, event):

        ev_weight = event.event_weight
        pt_hat = event.pt_hat
        vertex_x = event.vertex_x
        vertex_y = event.vertex_y
        vertex_xy_squared = np.sqrt(vertex_x*vertex_x + vertex_y*vertex_y)
        #angle_vertex_xy = np.arctan(vertex_y / vertex_x)
        #vertex_vec = np.array([vertex_x, vertex_y])
        vertex_vec = [vertex_x, vertex_y]
        x_vec = [1, 0]

        getattr(self, 'hPtHat').Fill(pt_hat, ev_weight) 
        getattr(self, 'hVertexX').Fill(vertex_x, ev_weight) 
        getattr(self, 'hVertexY').Fill(vertex_y, ev_weight) 
        getattr(self, 'hVertexXY').Fill(vertex_x, vertex_y, ev_weight) 
        getattr(self, 'hPtHat').Fill(pt_hat, ev_weight) 

        max_angle = 10
    
        # Loop through hadrons
        for hadron in hadrons:

            # Fill some basic hadron info
            pt = hadron.pt()

            px = hadron.px()
            py = hadron.py()
            #p_vec = np.array([px, py])
            p_vec = [px, py]

            #angle_hadron_xy = np.arctan(py / px)
            #angle_vertex_hadron = angle_between(p_vec, vertex_vec) * 180 / np.pi
            #angle_vertex_hadron = py_ang(p_vec,vertex_vec) * 180 / np.pi
            angle_vertex_hadron = angle(p_vec,vertex_vec) * 180 / np.pi
            angle_x_hadron = angle(p_vec,x_vec) * 180 / np.pi

            #print('angle of vertex = ', angle_vertex_xy, ' angle of hadron = ',angle_hadron_xy, 'angle between hadron and vertex = ',angle_vertex_hadron)
            #print('angle between hadron and vertex = ',angle_vertex_hadron)

            # Fill charged hadron histograms (pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
            getattr(self, 'hChHadronPt').Fill(pt, ev_weight) 
            getattr(self, 'hSumVertexXYPtHat').Fill(vertex_xy_squared, pt_hat, ev_weight) 
            getattr(self, 'hSumVertexXYPtChHadron').Fill(vertex_xy_squared, pt, ev_weight) 
            getattr(self, 'hPtHatPtChHadron').Fill(pt_hat, pt, ev_weight) 
            getattr(self, 'hSumVertexXYPtHatPtChHadron').Fill(vertex_xy_squared,pt_hat, pt, ev_weight) 

            if angle_vertex_hadron < max_angle:
                getattr(self, 'hSumVertexXYPtChHadronDirected').Fill(vertex_xy_squared, pt, ev_weight) 
            if angle_x_hadron < max_angle and abs(vertex_y) < 2:
                getattr(self, 'hVertexXPtChHadronDirected').Fill(vertex_x, pt, ev_weight) 


            # rotate axes such that particle travelling parallel to x axis
            if px > 0:
                px_rot = math.sqrt(px*px + py*py)
            else:
                px_rot = -math.sqrt(px*px + py*py)

            py_rot = 0.
            angle_rot = math.atan(py/px)
            vertex_x_rot = vertex_x * math.cos(angle_rot) + vertex_y * math.sin(angle_rot)
            vertex_y_rot = -vertex_x * math.sin(angle_rot) + vertex_y * math.cos(angle_rot)
            # print(' - non-rotated - px = ', px,' py = ',py,' vertex_x = ',vertex_x,' vertex_y = ', vertex_y)
            # print('     - rotated - px = ', px_rot,' py = ',py_rot, ' vertex_x = ',vertex_x_rot,' vertex_y = ', vertex_y_rot, ' angle_rot = ',angle_rot)
            if abs(vertex_y_rot) < 2:
                getattr(self, 'hVertexXPtChHadronRotate').Fill(vertex_x_rot, px_rot, ev_weight) 


    # ---------------------------------------------------------------
    # Fill jet observables
    # For AA, we find three different collections of jets:
    #
    #   (1) Using shower+recoil particles, with constituent subtraction
    #        - No further hole subtraction necessary
    #
    #   (2) Using shower+recoil particles, using standard recombiner
    #       In this case, observable-specific hole subtraction necessary
    #       We consider three different classes of jet observables:
    #        (i) Jet pt-like observables -- subtract holes within R
    #        (ii) Additive substructure -- subtract holes within R
    #        (iii) Non-additive substructure -- correct the jet pt only
    #       We also save unsubtracted histograms for comparison.
    #
    #   (3) Using shower+recoil+hole particles, using negative recombiner
    #       In this case, observable-specific hole subtraction necessary
    #       We consider three different classes of jet observables:
    #        (i) Jet pt-like observables -- no further hole subtraction
    #        (ii) Additive substructure -- subtract holes within R
    #        (iii) Non-additive substructure -- we do no further hole subtraction
    # ---------------------------------------------------------------
    def fill_jet_observables(self, event, hadrons_positive, hadrons_negative,
                             hadrons_positive_charged, hadrons_negative_charged,
                             pid_hadrons_positive, pid_hadrons_negative,
                             pid_hadrons_positive_charged, pid_hadrons_negative_charged,
                             jet_collection_label=''):

        ev_weight = event.event_weight
        pt_hat = event.pt_hat
        vertex_x = event.vertex_x
        vertex_y = event.vertex_y

        # Set the appropriate lists of hadrons to input to the jet finding
        if jet_collection_label in ['', '_shower_recoil', '_constituent_subtraction']:
            hadrons_for_jet_finding = hadrons_positive
            hadrons_for_jet_finding_charged = hadrons_positive_charged
        elif jet_collection_label in ['_negative_recombiner']:
            hadrons_for_jet_finding = list(hadrons_positive) + list(hadrons_negative)
            hadrons_for_jet_finding_charged = list(hadrons_positive_charged) + list(hadrons_negative_charged)

        # Loop through specified jet R
        for jetR in self.jet_R:

            # Set jet definition and a jet selector
            jet_def = fj.JetDefinition(fj.antikt_algorithm, jetR)
            if jet_collection_label in ['_negative_recombiner']:
                recombiner = fjext.NegativeEnergyRecombiner()
                jet_def.set_recombiner(recombiner)
            jet_selector = fj.SelectorPtMin(self.min_jet_pt) & fj.SelectorAbsRapMax(self.max_jet_y)


            # Fill jets
            cs = fj.ClusterSequence(hadrons_for_jet_finding, jet_def)
            jets = fj.sorted_by_pt(cs.inclusive_jets())
            jets_selected = jet_selector(jets)

            # loop over jets
            for jet in jets_selected:

                #   For the shower+recoil case, we need to subtract the hole pt
                #   For the negative recombiner case, we do not need to adjust the pt, but we want to keep track of the holes
                holes_in_jet = []
                if jet_collection_label in ['_shower_recoil', '_negative_recombiner']:
                    for hadron in hadrons_negative:
                        if jet.delta_R(hadron) < jetR:
                            holes_in_jet.append(hadron)
    
                # Correct the pt of the jet, if applicable
                # For pp or negative recombiner or constituent subtraction case, we do not need to adjust the pt
                # For the shower+recoil case, we need to subtract the hole pt
                if jet_collection_label in ['', '_negative_recombiner', '_constituent_subtraction']:
                    jet_pt = jet_pt_uncorrected = jet.pt()
                    jet_px = jet_px_uncorrected = jet.px()
                    jet_py = jet_py_uncorrected = jet.py()
                elif jet_collection_label in ['_shower_recoil']:
                    negative_pt = 0.
                    negative_px = 0.
                    negative_py = 0.
                    for hadron in holes_in_jet:
                        negative_pt += hadron.pt()
                        negative_px += hadron.px()
                        negative_py += hadron.py()
                    jet_pt_uncorrected = jet.pt()               # uncorrected pt: shower+recoil
                    jet_px_uncorrected = jet.px()               # uncorrected pt: shower+recoil
                    jet_py_uncorrected = jet.py()               # uncorrected pt: shower+recoil
                    jet_pt = jet_pt_uncorrected - negative_pt   # corrected pt: shower+recoil-holes
                    jet_px = jet_px_uncorrected - negative_px   # corrected pt: shower+recoil-holes
                    jet_py = jet_py_uncorrected - negative_py   # corrected pt: shower+recoil-holes
                    # rotate
                    jet_px_rot, jet_py_rot, vertex_x_rot, vertex_y_rot = rotate_axes(jet_px, jet_py, vertex_x, vertex_y)

                    if abs(vertex_y_rot) < 2:

                        hname = f'hVertexXPtJetRotate_{jetR}{jet_collection_label}'
                        getattr(self, hname).Fill(vertex_x_rot, jet_px_rot, ev_weight) 


                    #print('corr  : jet pt = {:.2f} jet px = {:.2f} jet py = {:.2f} jet pt from xy = {:.2f}'.format(jet_pt, jet_px, jet_py, math.sqrt(jet_px*jet_px + jet_py*jet_py)))
                    #print('uncorr: jet pt = {:.2f} jet px = {:.2f} jet py = {:.2f}'.format(jet_pt_uncorrected, jet_px_uncorrected, jet_py_uncorrected))
                    #print('not rotated: jet px = {:.2f} jet py = {:.2f} vertex x = {:.2f} vertex y = {:.2f}'.format(jet_px, jet_py, vertex_x, vertex_y))
                    #print('rotated:     jet px = {:.2f} jet py = {:.2f} vertex x = {:.2f} vertex y = {:.2f}'.format(jet_px_rot, jet_py_rot, vertex_x_rot, vertex_y_rot))

    




def rotate_axes(px, py, vertex_x, vertex_y):

    # rotate axes such that particle travelling parallel to x axis
    if px > 0:
        px_rot = math.sqrt(px*px + py*py)
    else:
        px_rot = -math.sqrt(px*px + py*py)

    py_rot = 0.
    angle_rot = math.atan(py/px)
    vertex_x_rot = vertex_x * math.cos(angle_rot) + vertex_y * math.sin(angle_rot)
    vertex_y_rot = -vertex_x * math.sin(angle_rot) + vertex_y * math.cos(angle_rot)

    return px_rot, py_rot, vertex_x_rot, vertex_y_rot





def dotproduct(v1, v2):
      return sum((a*b) for a, b in zip(v1, v2))

def length(v):
      return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
      return math.acos(dotproduct(v1, v2) / ((length(v1) * length(v2))+0.00000001))

def py_ang(v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'    """
        cosang = np.dot(v1, v2)
        sinang = np.linalg.norm(np.cross(v1, v2))
        return np.arctan2(sinang, cosang)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle between two vectors.  """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    # ---------------------------------------------------------------
    # Compute electric charge from pid
    # ---------------------------------------------------------------
    def charge(self, pid):

        if pid in [11, 13, -211, -321, -2212, -3222, 3112, 3312, 3334]:
            return -1.
        elif pid in [-11, -13, 211, 321, 2212, 3222, -3112, -3312, -3334]:
            return 1.
        elif pid in [22, 111, 2112]:
            return 0.
        else:
            sys.exit(f'failed to compute charge of pid {pid}')

##################################################################
if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser(description="Generate JETSCAPE events")
    parser.add_argument(
        "-c",
        "--configFile",
        action="store",
        type=str,
        metavar="configFile",
        default="/home/jetscape-user/JETSCAPE-analysis/config/jetscapeAnalysisConfig.yaml",
        help="Path of config file for analysis",
    )
    parser.add_argument(
        "-i",
        "--inputFile",
        action="store",
        type=str,
        metavar="inputDir",
        default="/home/jetscape-user/JETSCAPE-analysis/test.out",
        help="Input directory containing JETSCAPE output files",
    )
    parser.add_argument(
        "-o",
        "--outputDir",
        action="store",
        type=str,
        metavar="outputDir",
        default="/home/jetscape-user/JETSCAPE-analysis/TestOutput",
        help="Output directory for output to be written to",
    )

    # Parse the arguments
    args = parser.parse_args()

    # If invalid configFile is given, exit
    if not os.path.exists(args.configFile):
        print('File "{0}" does not exist! Exiting!'.format(args.configFile))
        sys.exit(0)

    # If invalid inputDir is given, exit
    if not os.path.exists(args.inputFile):
        print('File "{0}" does not exist! Exiting!'.format(args.inputFile))
        sys.exit(0)

    analysis = AnalyzeJetscapeEvents_STAT(config_file=args.configFile, input_file=args.inputFile, output_dir=args.outputDir)
    analysis.analyze_jetscape_events()
