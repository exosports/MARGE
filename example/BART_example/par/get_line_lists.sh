#! /bin/bash

echo "Downloading HITRAN/HITEMP line lists..."
wget --user=HITRAN --password=getdata -N -i wget-list_HITEMP-CO2.txt
wget --user=HITRAN --password=getdata -N -i wget-list_HITEMP-CO.txt
wget --user=HITRAN --password=getdata -N -i wget-list_HITEMP-H2O.txt
wget --user=HITRAN --password=getdata -N -i wget-list_HITRAN-CH4.txt
wget --user=HITRAN --password=getdata -N -i wget-list_HITRAN-H2.txt
echo "Extracting archives..."
unzip '01_*HITEMP2010.zip'
unzip '02_*HITEMP2010.zip'
unzip '05_*HITEMP2010.zip'
unzip '06_hit12.zip'
unzip '45_hit12.zip'
rm -f *.zip
echo "Finished retrieving HITRAN/HITEMP line lists."
