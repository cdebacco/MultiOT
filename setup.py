import os

packages = ["jupyterlab",
	"notebook",
	"matplotlib",
	"pandas",
	"networkx",
	"scipy",
	"libpysal",
	"pysal"]

for st_ in packages:
	os.system("pip install "+st_)

os.system("python3 -m pip install -U --pre shapely")
os.system("pip install geopandas")
