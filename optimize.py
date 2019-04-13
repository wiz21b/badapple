""" This is quick and dirty
Norallly one should give the kmean algorithm the proper
init parameters. When you don't do it, it chooses them
at random, giving varying results.

So, instead of looking at the best init params, I just
run kmeans dozens of time to find the best outcome.
That just brute force.

"""

import subprocess
import shutil
import os

def d(p):
    if os.path.exists(p):
        os.remove(p)


def s(p):
    if os.path.exists(p):
        r = os.stat(p).st_size
        print("Size of {} is {}".format(p,r))
        return r
    else:
        return 0

d("data.a.best")
d("cstripes.data.best")

for i in range(100):
    subprocess.run("python kmean.py")
    print("-"*80 + " Run {}".format(i))
    if s("cstripes.data") < (s("cstripes.data.best") or 100000000):
        print("Best shot {}".format(s("cstripes.data")))
        shutil.copyfile("data.a", "data.a.best")
        shutil.copyfile("cstripes.data", "cstripes.data.best")
    else:
        print("Not better")

shutil.copyfile("data.a.best", "data.a")
shutil.copyfile("cstripes.data.best", "cstripes.data")
