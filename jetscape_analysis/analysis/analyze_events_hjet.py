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

    #    # Load observable blocks
    #    self.hadron_observables = config['hadron']
    #    self.hadron_correlation_observables = config['hadron_correlations']
    #    self.inclusive_chjet_observables = config['inclusive_chjet']
    #    self.inclusive_jet_observables = {}
    #    self.semi_inclusive_chjet_observables = {}
    #    self.dijet_observables = {}
    #    if 'inclusive_jet' in config:
    #        self.inclusive_jet_observables = config['inclusive_jet']
    #    if 'semi_inclusive_chjet' in config:
    #        self.semi_inclusive_chjet_observables = config['semi_inclusive_chjet']
    #    if 'dijet' in config:
    #        self.dijet_observables = config['dijet']

    #    # General jet finding parameters
    #    self.jet_R = config['jet_R']
    #    self.min_jet_pt = config['min_jet_pt']
    #    self.max_jet_y = config['max_jet_y']

    #    # General grooming parameters'
    #    self.grooming_settings = {}
    #    if 'SoftDrop' in config:
    #        self.grooming_settings = config['SoftDrop']

    #    # If AA, set different options for hole subtraction treatment
    #    if self.is_AA:
    #        self.jet_collection_labels = config['jet_collection_labels']
    #    else:
    #        self.jet_collection_labels = ['']

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

        # Create list of charged particles
        fj_hadrons_positive_charged, pid_hadrons_positive_charged = self.fill_fastjet_constituents(event, select_status='+',
                                                                     select_charged=True)

        self.fill_hadron_histograms_fj(fj_hadrons_positive_charged, event)


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

        getattr(self, 'hPtHat').Fill(pt_hat, ev_weight) 
        getattr(self, 'hVertexX').Fill(vertex_x, ev_weight) 
        getattr(self, 'hVertexY').Fill(vertex_y, ev_weight) 
        getattr(self, 'hVertexXY').Fill(vertex_x, vertex_y, ev_weight) 
        getattr(self, 'hPtHat').Fill(pt_hat, ev_weight) 
    
        # Loop through hadrons
        for hadron in hadrons:

            # Fill some basic hadron info
            pt = hadron.pt()

            # Fill charged hadron histograms (pi+, K+, p+, Sigma+, Sigma-, Xi-, Omega-)
            getattr(self, 'hChHadronPt').Fill(pt, ev_weight) 
            getattr(self, 'hSumVertexXYPtHat').Fill(vertex_xy_squared, pt_hat, ev_weight) 
            getattr(self, 'hSumVertexXYPtChHadron').Fill(vertex_xy_squared, pt, ev_weight) 
            getattr(self, 'hPtHatPtChHadron').Fill(pt_hat, pt, ev_weight) 
            getattr(self, 'hSumVertexXYPtHatPtChHadron').Fill(vertex_xy_squared,pt_hat, pt, ev_weight) 


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
