import os
import random

# variables
KB = 1024
MB = KB * 1024
GB = MB * 1024

if not os.path.isfile('mb64'):
  with open('mb64', 'wb') as fout:
    fout.write(os.urandom(64 * MB))
    print "64"

if not os.path.isfile('mb128'):
  with open('mb128', 'wb') as fout:
    fout.write(os.urandom(128 * MB))
    print "128"

if not os.path.isfile('mb256'):
  with open('mb256', 'wb') as fout:
    fout.write(os.urandom(256 * MB))
    print "256"

if not os.path.isfile('mb512'):
  with open('mb512', 'wb') as fout:
    fout.write(os.urandom(512 * MB))
    print "512"
	
if not os.path.isfile('mb768'):
  with open('mb768', 'wb') as fout:
    fout.write(os.urandom(786 * MB))
    print "786"

if not os.path.isfile('mb1024'):
  with open('mb1024', 'wb') as fout:
    fout.write(os.urandom(1024 * MB))
    print "1024"
