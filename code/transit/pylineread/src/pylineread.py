#!/usr/bin/env python

# Copyright (C) 2015-2016 University of Central Florida. All rights reserved.
# Transit is under an open-source, reproducible-research license (see LICENSE).

import matplotlib.pyplot as plt

import numpy as np
import scipy.constants as sc
import sys, os, time
import multiprocessing as mp

from six.moves import configparser
import six
if six.PY2:
    ConfigParser = configparser.SafeConfigParser
else:
    ConfigParser = configparser.ConfigParser
import argparse
import struct
import heapq as hq

import constants as c
import utils     as ut
import db_pands  as ps
import db_hitran as hit
import db_tioschwenke as ts
import db_voplez as vo


def parseargs():
  """
  Read and process the command line arguments.

  Returns:
  --------
  args: Namespace object
     Object with the command line arguments.

  Modification History:
  ---------------------
  2013        madison   Initial implementation.        
  2014-03-06  patricio  Updated from optparse to argparse. Added documentation
                        and config_file option.    pcubillos@fulbrightmail.org
  2017-10-27  mhimes    Added multiprocessing. Added ncores (number of cores) 
                        and defn (HITRAN molecules definition file) config args
  """
  # Parser to process a configuration file:
  cparser = argparse.ArgumentParser(description=__doc__, add_help=False,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
  # Add config file option:
  cparser.add_argument("-c", "--config_file",
                       help="Configuration filename (string).", metavar="FILE")
  # remaining_argv contains all other command-line-arguments:
  args, remaining_argv = cparser.parse_known_args()

  # Get parameters from configuration file (if exists):
  if args.config_file:
    if not os.path.isfile(args.config_file):
      ut.printexit("Configuration file '{:s}' does not exist.".
                    format(args.config_file))
    config = ConfigParser()
    config.read([args.config_file])
    if "Parameters" not in config.sections():
      ut.printexit("Invalid configuration file: '{:s}'.".
                    format(args.config_file))
    defaults = dict(config.items("Parameters"))
    # Store these arguments as lists:
    if "db_list" in defaults:
      defaults["db_list"]   = defaults["db_list"].split()
    if "part_list" in defaults:
      defaults["part_list"] = defaults["part_list"].split()
    if "dbtype" in defaults:
      defaults["dbtype"]    = defaults["dbtype"].split()
  else:
    defaults = {}

  # Inherit options from cparser:
  parser = argparse.ArgumentParser(parents=[cparser])

  # General Options:
  parser.add_argument("-v", "--verbose-level", action="store",
                       help="Verbosity level (integer) [default: %(default)s].",
                       dest="verb", type=int, default=2)
  parser.add_argument("-q", "--quiet",         action="store_false",
                       help="Set verbosity level to 0.",
                       dest="verb")
  parser.add_argument("-N", "--ncores", action="store", 
                      help="Maximum number of cores",
                      dest="ncores", type=int, default=1)
  # Database Options:
  group = parser.add_argument_group("Database Options")
  group.add_argument("-o", "--output",         action  = "store",
                     help="Output filename (string) [default: '%(default)s'].",
                     dest= "output",           default = "output.tli")
  group.add_argument("-d", "--database",       action="append",
                     help="Path (string) to the input line-transition "
                          "database file(s).",
                     dest="db_list")
  group.add_argument("-p", "--partition",      action="append",
                     help="Path (string) to the auxiliary partition-function "
                          "file(s).",
                     dest="part_list")
  group.add_argument("-t", "--dbtype",         action="append",
                     help="Database type (string).  'ps' for Partridge & "
                          "Schwenke's H2O; 'hit' for HITRAN and HITEMP; "
                          "'ts' for Schwenke's TiO, or 'vo' for Plez's VO.",
                     choices=('ps', 'hit', 'ts', 'vo'),
                     dest="dbtype")
  # HITRAN definitions file
  group.add_argument("-n", "--defn",           action="store", 
                     help="Path/to/file of HITRAN definitions file", 
                     dest="defn")
  # Wavelength Options:
  group = parser.add_argument_group("Wavelength Options")
  group.add_argument("-i", "--wl-init",       action="store",
                     help="Initial wavelength (microns) [default: "
                          "%(default)s].",
                     dest="iwav", type=float, default=1.0)
  group.add_argument("-f", "--wl-final",      action="store",
                     help="Final wavelength (microns) [default: %(default)s].",
                     dest="fwav", type=float, default=2.0)
  parser.set_defaults(**defaults)
  args = parser.parse_args(remaining_argv)

  return args


if __name__ == "__main__":
  """
  Prototype code for porting lineread to python with P&S DB support
  Multiple DB types forthcoming with rolling revisions

  Parameters:
  -----------

  Modification History:
  ---------------------
  2013-10-21  madison   Initial version based on P. Rojo's lineread C code.
                                                 madison.stemm@ucf.edu
  2014-03-05  patricio  Added documentation and updated Madison's code.
                                                 pcubillos@fulbrightmail.org
  2014-07-27  patricio  Updated to version 5.0
  2017-10-27  mhimes    Implemented multiprocessing, defn argument for HITRAN
  """

  # Process command-line-arguments:
  cla = parseargs()
  # Unpack parameters:
  verbose    = cla.verb
  ncores     = cla.ncores
  dblist     = cla.db_list
  pflist     = cla.part_list
  dbtype     = cla.dbtype 
  outputfile = cla.output
  defn       = cla.defn

  # Number of files:
  Nfiles = len(dblist)
  # Driver routine to read the databases:
  driver = []
  for i in np.arange(Nfiles):
    if   dbtype[i] == "ps":
      driver.append(ps.pands(dblist[i],       pflist[i]))
    elif dbtype[i] == "hit":
      driver.append(hit.hitran(dblist[i],     pflist[i], defn))
    elif dbtype[i] == "ts":
      driver.append(ts.tioschwenke(dblist[i], pflist[i]))
    elif dbtype[i] == "vo":
      driver.append(vo.voplez(dblist[i],      pflist[i]))
    else:
      ut.printexit("Unknown Database type ({:d}): '{:s}'".
                    format(i+1, dbtype[i]))
    ut.lrprint(verbose-10, "File {:d}, database name: '{:s}'".
                            format(i+1, driver[i].name))

  ut.lrprint(verbose, "Beginning to write the TLI file: '{:s}'".
                       format(outputfile))
  # Open output file:
  TLIout  = open(outputfile, "wb")

  # Get the machine endian type (big/little):
  endianness = sys.byteorder

  # Hardcoded implementation of lineread's "magic number" check for endianness
  # derived from binary encoding the letters TLI into binary location
  # and checking the order
  if endianness == 'big':
    magic = b'\xab\xb3\xb6\xff'
  if endianness == 'little':
    magic = b'\xff\xb6\xb3\xab'

  # TLI header: Tells endianness of binary, TLI version, and number of
  # databases used.
  header = magic
  header += struct.pack("3h", c.TLI_VERSION, c.LR_VERSION, c.LR_REVISION)
  # Add initial and final wavelength boundaries:
  header += struct.pack("2d", cla.iwav, cla.fwav)

  # Get number of databases:
  DBnames = [] # Database names
  DBskip  = [] # Index of repeated databases
  for i in np.arange(Nfiles):
    dbname = driver[i].name
    if dbname in DBnames:
      DBskip.append(i) # Ommit repeated databases
    else:
      DBnames.append(dbname) 
  Ndb = len(DBnames)
  header += struct.pack("h", Ndb)
  TLIout.write(header)

  ut.lrprint(verbose-8, "Magic number: %d\n"
                        "TLI Version:  %d\n"
                        "LR Version:   %d\n"
                        "LR Revision:  %d\n"
                        "Initial wavelength (um): %7.3f\n"
                        "Final   wavelength (um): %7.3f"%(
                         struct.unpack('i', magic)[0],
                         c.TLI_VERSION, c.LR_VERSION, c.LR_REVISION,
                         cla.iwav, cla.fwav))
  ut.lrprint(verbose-8, "There are {:d} databases in {:d} files.".
                         format(Ndb, Nfiles))
  ut.lrprint(verbose-9, "Databases:\n{}".format(DBnames))

  # Partition info:
  totIso = 0                  # Cumulative number of isotopes
  acum = np.zeros(Ndb+1, int) # Cumul. number of isotopes per database

  ut.lrprint(verbose-2, "Reading and writting partition function info.")
  # Database correlative number:
  idb = 0
  # Loop through the partition files (if more than one) and write the
  # data to a processed TLI file:
  for i in np.arange(Nfiles):
    # Skip if we already stored the PF info of this DB:
    if i in DBskip:
      continue

    # Get partition function values:
    Temp, partDB = driver[i].getpf(verbose)
    isoNames  = driver[i].isotopes
    iso_mass  = driver[i].mass
    iso_ratio = driver[i].isoratio

    # Number of temperature samples:
    Ntemp = len(Temp)
    # Number of isotopes:
    Niso  = len(isoNames)

    # Get file name (remove root and extension):
    lenDBname = len(DBnames[idb])
    lenMolec  = len(driver[i].molecule)

    # Store length of database name, database name, number of temperatures,
    #  and number of isotopes in TLI file:
    TLIout.write(struct.pack("h{:d}B".format(lenDBname),
                             lenDBname, *DBnames[idb].encode()))
    # Store the molecule name:
    TLIout.write(struct.pack("h{:d}B".format(lenMolec),
                             lenMolec, *driver[i].molecule.encode()))
    # Store the number of temperature samples and isotopes:
    TLIout.write(struct.pack("hh", Ntemp, Niso))
    ut.lrprint(verbose-8, "\nDatabase ({:d}/{:d}): '{:s} ({:s} molecule)'\n"
                          "  Number of temperatures: {:3d}\n"
                          "  Number of isotopes:     {:3d}".
                               format(idb+1, Ndb, DBnames[idb],
                                      driver[i].molecule, Ntemp, Niso))

    # Write all the temperatures in a list:
    TLIout.write(struct.pack("{:d}d".format(Ntemp), *Temp))
    ut.lrprint(verbose-8, "  Temperatures: [{:6.1f}, {:6.1f}, ..., {:6.1f}]".
                           format(Temp[0], Temp[1], Temp[Ntemp-1]))

    # For each isotope, write the relevant partition function information
    # to file.  Keep a tally of isotopes for multiple databse support 
    for j in np.arange(Niso):
      ut.lrprint(verbose-9, "  Isotope ({:d}/{:d}): '{:s}'".
                              format(j+1, Niso, isoNames[j]))

      # Store length of isotope name, isotope name, and isotope mass:
      lenIsoname = len(isoNames[j])
      Iname = str(isoNames[j])
      TLIout.write(struct.pack("h{:d}B".format(lenIsoname), lenIsoname, *Iname.encode()))
      TLIout.write(struct.pack("d", iso_mass[j]))
      TLIout.write(struct.pack("d", iso_ratio[j]))

      # Write the partition function per isotope:
      TLIout.write(struct.pack("{:d}d".format(Ntemp), *partDB[j]))
      ut.lrprint(verbose-9, "    Mass (u):        {:8.4f}\n"
                          "    Isotopic ratio:  {:8.4g}\n"
                          "    Part. Function:  [{:.2e}, {:.2e}, ..., {:.2e}]".
                            format(iso_mass[j], iso_ratio[j],
                                   partDB[j,0], partDB[j,1], partDB[j,Ntemp-1]))

    # Calculate cumulative number of isotopes per database:
    totIso += Niso
    idb += 1
    acum[idb] = totIso

  # Cumulative number of isotopes:
  ut.lrprint(verbose-5, "Cumulative number of isotopes per DB: {}".format(acum))
  ut.lrprint(verbose, "Done.")

  # Initialize arrays to be filled
  ut.lrprint(verbose, "\nWriting transition info to tli file:")
  wlength = np.array([], np.double)
  gf      = np.array([], np.double)
  elow    = np.array([], np.double)
  isoID   = np.array([], int)

  # Initialize these for multiprocessing
  jobs = []
  manager = mp.Manager()
  return_dict = manager.dict()

  ti = time.time()
  
  # Read from file and write the transition info:
  for db in np.arange(Nfiles):
    # Read databases:
    p = mp.Process(target = driver[db].dbread, args = (cla.iwav, cla.fwav,
                                                       cla.verb, 
                                                       db, return_dict,
                                                       pflist[db]))
    jobs.append(p)
    p.start()

    # Limit number of running processes
    while True:
      procs = 0
      # Count the running processes
      for proc in jobs:
        if proc.is_alive():
          procs += 1
      # If less processes running than the limit, exit loop to start more
      if procs <= ncores:
        break
      # Chill for a sec
      time.sleep(.1)

  # Join the results together
  for proc in jobs:
      proc.join()
      
  # Print out the time it took to read the files
  tf = time.time()
  ut.lrprint(verbose-3, "Total reading time: {:8.3f} seconds".format(tf-ti))

  transDB = return_dict

  # Store the info
  for db in np.arange(Nfiles):
    # Get database index:
    dbname = driver[db].name
    idb = DBnames.index(dbname)

    # Combine the results into 1 array
    wlength = np.concatenate((wlength, transDB[db][0]))    
    gf      = np.concatenate((gf,      transDB[db][1]))    
    elow    = np.concatenate((elow,    transDB[db][2]))    
    isoID   = np.concatenate((isoID,   transDB[db][3]+acum[idb]))    

    ut.lrprint(verbose-8, "Isotope in-database indices: {}".format(
                           np.unique(transDB[db][3])))
    ut.lrprint(verbose-8, "Isotope correlative indices: {}\n".format(
                           np.unique(transDB[db][3]+acum[idb])))

  # Total number of transitions:
  nTransitions = np.size(wlength)


  # Calculate the total number of transitions per isotope:
  Nisotran = np.bincount(isoID)
  Nisotran = Nisotran[np.where(Nisotran>0)]  # Remove zeroes
  ut.lrprint(verbose-5, "Transitions per isotope:\n{}".format(Nisotran))

  # Sort the line transitions (by isotope, then by wavelength):
  ti = time.time()
  #isort = sorted(zip(np.arange(nTransitions), isoID, wlength),
  #               key=lambda x:(x[1], x[2]))
  #isort = list(zip(*isort)[0])
  isort = np.argsort(isoID)
  # Sort each isotope by wavenumber:
  ihi = 0
  for j in np.arange(len(Nisotran)):
    ilo  = ihi
    ihi += Nisotran[j]
    wlsort = np.argsort(wlength[isort][ilo:ihi])
    isort[ilo:ihi] = isort[ilo:ihi][wlsort]

  tf = time.time()
  ut.lrprint(verbose-3, "Sort time:    {:8.3f} seconds".format(tf-ti))
  wlength = wlength[isort]
  gf      = gf     [isort]
  elow    = elow   [isort]
  isoID   = isoID  [isort]

  # FINDME: Implement well this:
  if False:
    plt.figure(0)  
    plt.clf()
    plt.plot(isoID)
    plt.xlabel("Line index")
    plt.ylabel("Isotope ID")
    plt.savefig("ID.png")

    plt.clf()
    plt.plot(wlength)
    plt.xlabel("Line index")
    plt.ylabel("Wavelength  (um)")
    plt.savefig("wavelength.png")

  # Pack:
  transinfo = b""
  # Write the number of transitions:
  TLIout.write(struct.pack("i", nTransitions))
  ut.lrprint(verbose-3, "Writing {:d} transition lines.".format(nTransitions))
  # Write the number of transitions for each isotope:
  nIso = len(Nisotran)
  # Note that nIso may differ from accumiso, since accum iso accounts for
  # all the existing isotopes for an species, whereas nIso accounts only
  # for the isotopes that do have line transitions in the given range.
  TLIout.write(struct.pack("i",nIso))
  TLIout.write(struct.pack(str(nIso)+"i", *list(Nisotran)))

  # Write the Line-transition data:
  ti = time.time()
  transinfo += struct.pack("{:d}d".format(nTransitions), *list(wlength))
  transinfo += struct.pack("{:d}h".format(nTransitions), *list(isoID))
  transinfo += struct.pack("{:d}d".format(nTransitions), *list(elow))
  transinfo += struct.pack("{:d}d".format(nTransitions), *list(gf))
  tf = time.time()
  ut.lrprint(verbose-3, "Packing time: {:8.3f} seconds".format(tf-ti))

  ti = time.time()
  TLIout.write(transinfo)
  tf = time.time()
  ut.lrprint(verbose-3, "Writing time: {:8.3f} seconds".format(tf-ti))

  TLIout.close()
  ut.lrprint(verbose, "Done.\n")
  sys.exit(0)

