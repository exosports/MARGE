# Copyright (C) 2015-2016 University of Central Florida. All rights reserved.
# Transit is under an open-source, reproducible-research license (see LICENSE).

#import db_pands  as ps
#import db_hitran as hit
#import db_tioschwenke as ts

class dbdriver(object):
  def __init__(self, dbfile, pffile):
    self.dbfile = dbfile
    self.pffile = pffile

  def getpf(self, verbose):
    """
      Calculate partition function for specific database type.

      Parameters:
      -----------
      dbtype: String
    """
    pass

  def dbread(self, iwl, fwl, verbose):
    """
      Read linelist values for specific database type.

      Parameters:
      -----------
      dbtype: String
    """
    pass

  def readwl(self, dbfile, irec):
    """
      Read the wavelength parameter as given in the database
    """
    pass

  def binsearch(self, dbfile, wavelength, ilo, ihi, searchup=True):
    """
    Do a binary search of record position in File dbfile that has wavelength iwl
 
    Parameters:
    -----------
    dbfile: File object
       File where to search.
    wavelength: Scalar
       Target wavelength (as given in each specific database).
    ilo: Integer
       Index of smallest record to search.
    ihi: Integer
       Index of largerst record to search.
    dbtype: String
       Database type: ['ps', 'hit']
    searchup: Boolean
       Search up (True) or down (False) the records for duplicate results
       after the binary search.
 
    Returns:
    --------
    Index of record with
 
    Modification History:
    ---------------------
    2013        madison   Initial implementation.
    2014-03-05  patricio  Added documentation, updated Madison's code, and
                          included searchup parameter
                                                    pcubillos@fulbrightmail.org
    """
    # imin and imax are the fixed boundaries where to search:
    imin, imax = ilo, ihi

    # Wavelength of record:
    rec_wl = 0
    irec   = 0
 
    # Start binary search:
    while ihi - ilo > 1:
      # Middle record index:
      irec = int((ihi + ilo) // 2)
 
      # Read wavelength, depending on linelist format:
      rec_wl = self.readwl(dbfile, irec)
      # Update search limits:
      if rec_wl > wavelength:
        ihi = irec
      else:
        ilo = irec
 
    # Start linear search:
    if searchup:
      irec = ilo
    else:
      irec = ihi

    # Check value of contiguous entries:
    icheck  = irec  # Record index to test
    bounded = True  # Record's wavelength is still within boundaries

    while bounded:
      # Update irec:
      irec = icheck
      # Check wavelength boundaries:
      if searchup:
        if irec == imax:  # Check index boundaries
          break
        icheck += 1
        bounded = self.readwl(dbfile, icheck) < wavelength
      else:
        if irec == imin:  # Check index boundaries
          break
        icheck -= 1
        bounded = self.readwl(dbfile, icheck) > wavelength

    # Move file pointer to begining:
    dbfile.seek(0)
 
    return irec

